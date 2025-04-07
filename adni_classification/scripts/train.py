"""Training script for ADNI classification."""

import os
import argparse
from typing import Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb

from adni_classification.models.model_factory import ModelFactory
from adni_classification.datasets.adni_dataset import ADNIDataset, get_transforms
from adni_classification.config import Config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ADNI classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g., configs/default.yaml)")
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    wandb_run: Optional[Any] = None,
) -> float:
    """Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        wandb_run: Weights & Biases run object for logging
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log batch metrics to wandb
        if wandb_run is not None and batch_idx % 10 == 0:
            wandb_run.log({
                "batch_loss": loss.item(),
                "batch": batch_idx,
            })
    
    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), correct / total


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
    
    # Create datasets
    train_dataset = ADNIDataset(
        csv_path=config.data.train_csv_path,
        img_dir=config.data.img_dir,
        transform=get_transforms("train"),
    )
    
    val_dataset = ADNIDataset(
        csv_path=config.data.val_csv_path,
        img_dir=config.data.img_dir,
        transform=get_transforms("val"),
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
    
    # Add pretrained parameters if specified
    if config.model.pretrained:
        model_kwargs["pretrained"] = True
        model_kwargs["weights_path"] = config.model.weights_path
    
    model = ModelFactory.create_model(
        model_name=config.model.name,
        **model_kwargs
    ).to(device)
    
    # Log model architecture to wandb
    if wandb_run is not None:
        wandb_run.watch(model, log="all")
    
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
        patience=5,
        verbose=True,
    )
    
    # Create output directory with run name
    output_dir = os.path.join(config.training.output_dir, config.wandb.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, wandb_run)
        print(f"Training Loss: {train_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({
                "train/epoch": epoch + 1,
                "train/loss": train_loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            })
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Log metrics to wandb
        if wandb_run is not None:
            wandb_run.log({
                "val/epoch": epoch + 1,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/learning_rate": optimizer.param_groups[0]["lr"],
            })
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to: {model_path}")

    # Close wandb run
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main() 