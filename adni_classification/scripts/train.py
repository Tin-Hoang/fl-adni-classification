"""Training script for ADNI classification."""

import os
import argparse
from typing import Any, Optional, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from adni_classification.models.model_factory import ModelFactory
from adni_classification.datasets.adni_dataset import ADNIDataset, get_transforms
from adni_classification.config import Config
from adni_classification.utils.visualization import visualize_batch, visualize_predictions, plot_training_history


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ADNI classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g., configs/default.yaml)")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples and predictions")
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    wandb_run: Optional[WandbRun] = None,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch
        wandb_run: Weights & Biases run object (optional)

    Returns:
        Tuple of (average training loss for the epoch, average training accuracy for the epoch)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Get data
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    # Log to wandb if enabled
    if wandb_run:
        wandb_run.log({
            "train/loss": avg_loss,
            "train/accuracy": accuracy,
        })

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    wandb_run: Optional[WandbRun] = None,
) -> tuple[float, float]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        wandb_run: Weights & Biases run object (optional)

    Returns:
        Tuple of (average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Get data
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    # Log to wandb if enabled
    if wandb_run:
        wandb_run.log({
            "val/loss": avg_loss,
            "val/accuracy": accuracy,
        })

    return avg_loss, accuracy


def main():
    """Main training function."""
    # Parse arguments and load config
    args = parse_args()
    config = Config.from_yaml(args.config)

    # Add timestamp to run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.wandb.run_name = f"{config.wandb.run_name}_{timestamp}"

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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = ADNIDataset(
        csv_path=config.data.train_csv_path,
        img_dir=config.data.img_dir,
        transform=get_transforms(
            "train",
            resize_size=config.data.resize_size,
            resize_mode=config.data.resize_mode
        ),
    )

    val_dataset = ADNIDataset(
        csv_path=config.data.val_csv_path,
        img_dir=config.data.img_dir,
        transform=get_transforms(
            "val",
            resize_size=config.data.resize_size,
            resize_mode=config.data.resize_mode
        ),
    )

    # Visualize samples if requested
    if args.visualize:
        print("Visualizing training samples...")
        visualize_batch(
            DataLoader(train_dataset, batch_size=min(5, len(train_dataset)), shuffle=True),
            save_path=os.path.join(config.training.output_dir, "train_samples.png")
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
    }

    # Add model-specific parameters
    if config.model.name == "resnet3d":
        model_kwargs["model_depth"] = config.model.model_depth
    elif config.model.name == "densenet3d":
        model_kwargs["growth_rate"] = config.model.growth_rate
        model_kwargs["block_config"] = config.model.block_config
    elif config.model.name == "simple3dcnn":
        model_kwargs["dropout_rate"] = 0.5  # Higher dropout for small dataset

    # Add pretrained parameters if specified
    if config.model.pretrained:
        model_kwargs["pretrained"] = True
        model_kwargs["weights_path"] = config.model.weights_path

    # Create model
    model = ModelFactory.create_model(config.model.name, **model_kwargs)
    model = model.to(device)

    # Print model summary
    print(f"Model: {config.model.name}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=True,
    )

    # Create output directory
    output_dir = os.path.join(config.training.output_dir, config.wandb.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    for epoch in range(config.training.num_epochs):
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, wandb_run
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, wandb_run)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pth"),
            )
            print(f"Saved best model with validation loss: {val_loss:.4f}")

        # Visualize predictions if requested
        if args.visualize and (epoch + 1) % 10 == 0:
            print(f"Visualizing predictions at epoch {epoch+1}...")
            visualize_predictions(
                model, val_loader, device,
                save_path=os.path.join(output_dir, f"predictions_epoch_{epoch+1}.png")
            )

    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(output_dir, "training_history.png")
    )

    print("Training complete!")

    # Close wandb run if enabled
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
