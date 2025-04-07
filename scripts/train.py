"""Training script for ADNI classification."""

import os
import argparse
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from wandb.sdk.wandb_run import Run as WandbRun
from torch.cuda.amp import autocast, GradScaler

from adni_classification.models.model_factory import ModelFactory
from adni_classification.datasets.adni_dataset import ADNIDataset, get_transforms
from adni_classification.utils.visualization import visualize_batch, visualize_predictions, plot_training_history
from adni_classification.config.config import Config


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    wandb_run: Optional[WandbRun] = None,
    log_batch_metrics: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: Optional[GradScaler] = None,
    use_mixed_precision: bool = False
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        wandb_run: Weights & Biases run object (optional)
        log_batch_metrics: Whether to log batch-level metrics
        gradient_accumulation_steps: Number of steps to accumulate gradients
        scaler: GradScaler for mixed precision training
        use_mixed_precision: Whether to use mixed precision training

    Returns:
        Tuple of (average training loss for the epoch, average training accuracy for the epoch)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Mixed precision training
        if use_mixed_precision and scaler is not None:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Scale gradients and backpropagate
            scaler.scale(loss).backward()

            # Step optimizer if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Regular training
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Step optimizer if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if log_batch_metrics and wandb_run is not None:
            wandb_run.log({
                "train/batch_loss": loss.item() * gradient_accumulation_steps,
                "train/batch_accuracy": 100.0 * correct / total,
            })

    # Handle any remaining gradients
    if use_mixed_precision and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = 100.0 * correct / total

    return avg_loss, avg_accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = False
) -> Tuple[float, float]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        use_mixed_precision: Whether to use mixed precision training

    Returns:
        Tuple of (average validation loss, average validation accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if use_mixed_precision:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = 100.0 * correct / total

    return avg_loss, avg_accuracy


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train ADNI classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples and predictions")
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if enabled
    wandb_run = None
    if config.wandb.use_wandb:
        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=config.wandb.tags,
            notes=config.wandb.notes,
            name=config.wandb.run_name,
            config=config.to_dict(),
        )

    # Create datasets with transforms
    train_transform = get_transforms(
        mode="train",
        resize_size=tuple(config.data.resize_size),
        resize_mode=config.data.resize_mode
    )

    val_transform = get_transforms(
        mode="val",
        resize_size=tuple(config.data.resize_size),
        resize_mode=config.data.resize_mode
    )

    train_dataset = ADNIDataset(
        csv_path=config.data.train_csv_path,
        img_dir=config.data.img_dir,
        transform=train_transform
    )

    val_dataset = ADNIDataset(
        csv_path=config.data.val_csv_path,
        img_dir=config.data.img_dir,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    # Create model
    model_kwargs = {
        "num_classes": config.model.num_classes,
        "pretrained": config.model.pretrained,
        "weights_path": config.model.weights_path,
    }

    # Add model-specific parameters
    if config.model.name == "resnet3d" and config.model.model_depth is not None:
        model_kwargs["model_depth"] = config.model.model_depth
    elif config.model.name == "densenet3d":
        if config.model.growth_rate is not None:
            model_kwargs["growth_rate"] = config.model.growth_rate
        if config.model.block_config is not None:
            model_kwargs["block_config"] = config.model.block_config

    model = ModelFactory.create_model(config.model.name, **model_kwargs)
    model = model.to(device)

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Initialize mixed precision training if enabled
    scaler = None
    use_mixed_precision = getattr(config.training, "mixed_precision", False)
    if use_mixed_precision:
        scaler = GradScaler('cuda')
        print("Using mixed precision training")

    # Get gradient accumulation steps
    gradient_accumulation_steps = getattr(config.training, "gradient_accumulation_steps", 1)
    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")

    # Visualize training samples if requested
    if args.visualize:
        print("Visualizing training samples...")
        visualize_batch(train_loader, num_samples=4, save_path=os.path.join(config.training.output_dir, "train_samples.png"))

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    for epoch in range(1, config.training.num_epochs + 1):
        print(f"Epoch {epoch}/{config.training.num_epochs}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, wandb_run,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_mixed_precision=use_mixed_precision
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if wandb_run is not None:
            wandb_run.log({
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                },
                step=epoch,
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(config.training.output_dir, f"{config.model.name}_best.pth"),
            )
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")

        # Visualize predictions every 10 epochs if requested
        if args.visualize and epoch % 10 == 0:
            print("Visualizing predictions...")
            visualize_predictions(
                model, val_loader, device, num_samples=4,
                save_path=os.path.join(config.training.output_dir, f"predictions_epoch_{epoch}.png")
            )

    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(config.training.output_dir, "training_history.png")
    )

    # Close wandb run
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
