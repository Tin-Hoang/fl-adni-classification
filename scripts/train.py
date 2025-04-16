"""Training script for ADNI classification."""

import os
import argparse
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from wandb.sdk.wandb_run import Run as WandbRun
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    ExponentialLR,
)

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

    # Create progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Mixed precision training
        if use_mixed_precision and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Scale gradients and backpropagate
            scaler.scale(loss).backward()

            # Step optimizer if we've accumulated enough gradients or it's the last batch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
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

            # Step optimizer if we've accumulated enough gradients or it's the last batch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        current_loss = total_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

        if log_batch_metrics and wandb_run is not None:
            wandb_run.log({
                "train/batch_loss": loss.item() * gradient_accumulation_steps,
                "train/batch_accuracy": 100.0 * correct / total,
            })

    # All gradient steps are handled in the loop now, no need for cleanup here
    avg_loss = total_loss / num_batches
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

    # Create progress bar
    pbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if use_mixed_precision:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = 100.0 * correct / total

    return avg_loss, avg_accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[GradScaler],
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    is_best: bool,
    output_dir: str,
    model_name: str,
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    checkpoint_config: Any
) -> None:
    """Save training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The learning rate scheduler state to save
        scaler: The GradScaler to save (if using mixed precision)
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        val_acc: Validation accuracy
        is_best: Whether this is the best model so far
        output_dir: Directory to save the checkpoint
        model_name: Name of the model for saving
        train_losses: History of training losses
        val_losses: History of validation losses
        train_accs: History of training accuracies
        val_accs: History of validation accuracies
        checkpoint_config: Configuration for checkpoint saving behavior
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }

    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add scaler state if using mixed precision
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Save regular checkpoint based on configuration
    if checkpoint_config.save_regular and epoch % checkpoint_config.save_frequency == 0:
        torch.save(
            checkpoint,
            os.path.join(output_dir, f"{model_name}_checkpoint_epoch_{epoch}.pth")
        )

    # Save latest checkpoint (overwrite) based on configuration
    if checkpoint_config.save_latest:
        torch.save(
            checkpoint,
            os.path.join(output_dir, f"{model_name}_checkpoint_latest.pth")
        )

    # Save best model if this is the best validation accuracy and enabled in configuration
    if is_best and checkpoint_config.save_best:
        print(f"New best model with validation accuracy: {val_acc:.2f}%")
        torch.save(
            checkpoint,
            os.path.join(output_dir, f"{model_name}_checkpoint_best.pth")
        )
        # Also save just the model state dict for compatibility
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{model_name}_best.pth")
        )


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Optional[GradScaler], int, List[float], List[float], List[float], List[float], float]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into (optional)
        scheduler: The learning rate scheduler to load state into (optional)
        scaler: The GradScaler to load state into (optional)

    Returns:
        Tuple of (model, optimizer, scheduler, scaler, start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc)
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)

    # Initialize history lists if they don't exist in the checkpoint
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])

    # Get the best validation accuracy
    best_val_acc = checkpoint.get('val_acc', 0.0)
    if val_accs:
        best_val_acc = max(best_val_acc, max(val_accs))

    return model, optimizer, scheduler, scaler, start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc


def get_scheduler(scheduler_type: str, optimizer: torch.optim.Optimizer, num_epochs: int) -> Optional[Any]:
    """Get the appropriate learning rate scheduler.

    Args:
        scheduler_type: Type of scheduler to use
        optimizer: Optimizer to use with the scheduler
        num_epochs: Total number of epochs

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        # Cosine annealing scheduler
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == "step":
        # Step scheduler (reduce LR by factor of gamma every step_size epochs)
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == "multistep":
        # Multi-step scheduler (reduce LR at specific milestones)
        return MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    elif scheduler_type == "plateau":
        # Reduce on plateau scheduler (reduce LR when validation metric plateaus)
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-6)
    elif scheduler_type == "exponential":
        # Exponential scheduler (reduce LR by gamma each epoch)
        return ExponentialLR(optimizer, gamma=0.95)
    else:
        # No scheduler
        print(f"[Warning] No learning rate scheduler specified, using no scheduler")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train ADNI classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Save the processed configuration to the output directory
    config_output_path = os.path.join(config.training.output_dir, "config.yaml")
    config.to_yaml(config_output_path)
    print(f"Saved configuration to {config_output_path}")
    print(f"Output directory: {config.training.output_dir}")

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

    # Set multiprocessing method to spawn for better cleanup
    if config.training.num_workers > 0:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # The method might already be set
            pass

    # Create datasets with transforms
    train_transform = get_transforms(
        mode="train",
        resize_size=tuple(config.data.resize_size),
        resize_mode=config.data.resize_mode,
        use_spacing=config.data.use_spacing,
        spacing_size=tuple(config.data.spacing_size)
    )

    val_transform = get_transforms(
        mode="val",
        resize_size=tuple(config.data.resize_size),
        resize_mode=config.data.resize_mode,
        use_spacing=config.data.use_spacing,
        spacing_size=tuple(config.data.spacing_size)
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

    # Create data loaders with proper multiprocessing settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.training.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.training.num_workers > 0 else False,
    )

    # Create model
    model_kwargs = {
        "num_classes": config.model.num_classes,
        "pretrained_checkpoint": config.model.pretrained_checkpoint,
    }

    # Add model-specific parameters
    if config.model.name == "resnet3d" and config.model.model_depth is not None:
        model_kwargs["model_depth"] = config.model.model_depth
    elif config.model.name == "densenet3d":
        if config.model.growth_rate is not None:
            model_kwargs["growth_rate"] = config.model.growth_rate
        if config.model.block_config is not None:
            model_kwargs["block_config"] = config.model.block_config

    # Pass data configuration for models that need it (like SecureFedCNN)
    if config.model.name == "securefed_cnn":
        model_kwargs["data"] = {
            "resize_size": config.data.resize_size
        }

    model = ModelFactory.create_model(config.model.name, **model_kwargs)
    model = model.to(device)

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create learning rate scheduler
    scheduler = get_scheduler(config.training.lr_scheduler, optimizer, config.training.num_epochs)
    print(f"Using learning rate scheduler: {config.training.lr_scheduler}")

    # Initialize mixed precision training if enabled
    scaler = None
    use_mixed_precision = getattr(config.training, "mixed_precision", False)
    if use_mixed_precision:
        scaler = GradScaler()
        print("Using mixed precision training")

    # Get gradient accumulation steps
    gradient_accumulation_steps = getattr(config.training, "gradient_accumulation_steps", 1)
    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")

    # Initialize training history variables
    start_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    # Check if we need to resume training from a checkpoint
    if config.model.pretrained_checkpoint:
        # Check if the weights path is a checkpoint file
        if os.path.isfile(config.model.pretrained_checkpoint):
            print(f"Loading checkpoint from: {config.model.pretrained_checkpoint}")
            model, optimizer, scheduler, scaler, start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc = load_checkpoint(
                config.model.pretrained_checkpoint, model, optimizer, scheduler, scaler
            )
            # We need to increment the epoch as start_epoch is the last completed epoch
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.2f}%")
        else:
            print(f"No checkpoint found at: {config.model.pretrained_checkpoint}")
            # If it's not a checkpoint file, it might be just a state dict
            if os.path.isfile(config.model.pretrained_checkpoint):
                print(f"Loading model state dict from: {config.model.pretrained_checkpoint}")
                state_dict = torch.load(config.model.pretrained_checkpoint, map_location=device)
                model.load_state_dict(state_dict)
                print(f"Loaded model state dict from: {config.model.pretrained_checkpoint}")

    # Visualize training samples if requested
    if config.training.visualize:
        print("Visualizing training samples...")
        visualize_batch(train_loader, num_samples=4, save_path=os.path.join(config.training.output_dir, "train_samples.png"))

    # Training loop
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

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

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)  # For ReduceLROnPlateau
            else:
                scheduler.step()  # For other schedulers

        # Log metrics to wandb
        if wandb_run is not None:
            wandb_log = {
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/lr": current_lr,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            }
            wandb_run.log(wandb_log, step=epoch + 1)

        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            is_best=is_best,
            output_dir=config.training.output_dir,
            model_name=config.model.name,
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            checkpoint_config=config.training.checkpoint
        )

        # Visualize predictions every 10 epochs if requested
        if config.training.visualize and (epoch + 1) % 10 == 0:
            print("Visualizing predictions...")
            visualize_predictions(
                model, val_loader, device, num_samples=4,
                save_path=os.path.join(config.training.output_dir, f"predictions_epoch_{epoch + 1}.png")
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
