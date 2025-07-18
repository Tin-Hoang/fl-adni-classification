"""Training script for ADNI classification."""

import argparse
import gc
import os
import tempfile
import time
from collections import Counter
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from adni_classification.config.config import Config
from adni_classification.datasets.dataset_factory import create_adni_dataset, get_transforms_from_config
from adni_classification.models.model_factory import ModelFactory
from adni_classification.utils.losses import create_loss_function
from adni_classification.utils.torch_utils import set_seed
from adni_classification.utils.training_utils import get_scheduler
from adni_classification.utils.visualization import (
    log_sample_images_to_wandb,
    plot_confusion_matrix,
    plot_training_history,
    visualize_batch,
    visualize_predictions,
)
from wandb.sdk.wandb_run import Run as WandbRun


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    weight_type: str = "inverse",
    manual_weights: Optional[List[float]] = None
) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Args:
        labels: List of class labels from the training set
        num_classes: Number of classes
        weight_type: Type of weighting to use ('inverse', 'sqrt_inverse', 'effective', 'manual')
        manual_weights: Manual weights to use if weight_type is 'manual'

    Returns:
        Tensor of class weights
    """
    # Get class frequencies
    class_counts = Counter(labels)

    # Ensure all classes are represented in the counts
    for c in range(num_classes):
        if c not in class_counts:
            class_counts[c] = 0

    # Sort the counts by class index
    sorted_counts = [class_counts[i] for i in range(num_classes)]
    total_samples = sum(sorted_counts)

    if weight_type == "inverse":
        # Inverse frequency weighting
        class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in sorted_counts]
    elif weight_type == "sqrt_inverse":
        # Square root of inverse frequency (less aggressive than inverse)
        class_weights = [
            np.sqrt(total_samples / (num_classes * count)) if count > 0 else 1.0
            for count in sorted_counts
        ]
    elif weight_type == "effective":
        # Effective number of samples weighting with beta=0.9999
        beta = 0.9999
        effective_nums = [1.0 - np.power(beta, count) for count in sorted_counts]
        class_weights = [(1.0 - beta) / num if num > 0 else 1.0 for num in effective_nums]
    elif weight_type == "manual" and manual_weights is not None:
        # Manually specified weights
        if len(manual_weights) != num_classes:
            raise ValueError(f"manual_weights must have length {num_classes}")
        class_weights = manual_weights
    else:
        # Default: all classes weighted equally
        class_weights = [1.0] * num_classes

    print(f"Class counts: {sorted_counts}")
    print(f"Class weights ({weight_type}): {class_weights}")

    return torch.FloatTensor(class_weights)


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
    batch_idx = -1

    # Add error handling for DataLoader issues
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

    # Calculate final metrics based on what was successfully processed
    avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0
    avg_accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, avg_accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        use_mixed_precision: Whether to use mixed precision training

    Returns:
        Tuple of (average validation loss, average validation accuracy, true labels, predicted labels)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # Create progress bar
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    batch_idx = -1

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

            # Collect true and predicted labels for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    # Calculate metrics based on what was successfully processed
    avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0
    avg_accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, avg_accuracy, np.array(all_labels), np.array(all_predictions)


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
    checkpoint_config: Any,
    class_weights: Optional[torch.Tensor] = None
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
        class_weights: Optional class weights used for loss function
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

    # Add class weights if they exist
    if class_weights is not None:
        checkpoint['class_weights'] = class_weights.cpu()

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
) -> Tuple[
    nn.Module,
    Optional[torch.optim.Optimizer],
    Optional[Any],
    Optional[GradScaler],
    int,
    List[float],
    List[float],
    List[float],
    List[float],
    float,
    Optional[torch.Tensor]
]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into (optional)
        scheduler: The learning rate scheduler to load state into (optional)
        scaler: The GradScaler to load state into (optional)

    Returns:
        Tuple of (model, optimizer, scheduler, scaler, start_epoch, train_losses,
                 val_losses, train_accs, val_accs, best_val_acc, class_weights)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e

    # Check if this is a proper training checkpoint format
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary. Got {type(checkpoint)}")

    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict'. Available keys: {list(checkpoint.keys())}")

    # Load model state dict with shape checking
    try:
        model_state_dict = checkpoint['model_state_dict']
        current_state_dict = model.state_dict()

        # Check for shape mismatches
        mismatched_keys = []
        for key, value in model_state_dict.items():
            if key in current_state_dict:
                if current_state_dict[key].shape != value.shape:
                    mismatched_keys.append(f"{key}: expected {current_state_dict[key].shape}, got {value.shape}")

        if mismatched_keys:
            print("Warning: Found shape mismatches in checkpoint:")
            for mismatch in mismatched_keys:
                print(f"  {mismatch}")
            print("Loading with strict=False to skip mismatched layers")
            model.load_state_dict(model_state_dict, strict=False)
        else:
            model.load_state_dict(model_state_dict)

        print("✓ Model state dict loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model state dict: {e}") from e

    # Load optimizer state if available
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}")

    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load scheduler state: {e}")

    # Load scaler state if available
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ Mixed precision scaler state loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load scaler state: {e}")

    # Extract training information
    start_epoch = checkpoint.get('epoch', 0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])

    # Get the best validation accuracy
    best_val_acc = checkpoint.get('val_acc', 0.0)
    if val_accs:
        best_val_acc = max(best_val_acc, max(val_accs))

    # Get class weights if they exist
    class_weights = checkpoint.get('class_weights', None)

    # Print checkpoint metadata if available
    if 'pretrained_info' in checkpoint:
        print("Checkpoint metadata:")
        for key, value in checkpoint['pretrained_info'].items():
            print(f"  {key}: {value}")

    print(f"✓ Checkpoint loaded: epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    return (
        model, optimizer, scheduler, scaler, start_epoch,
        train_losses, val_losses, train_accs, val_accs, best_val_acc, class_weights
    )


def cleanup_resources():
    """Clean up resources to prevent file descriptor leaks."""
    print("Cleaning up resources...")

    # Only perform critical cleanup operations
    # 1. Empty CUDA cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # 2. Run garbage collection once
    gc.collect()

    # 3. Only clean temp directories if we're experiencing file-related issues
    # This approach is much less resource-intensive
    try:
        # Check for too many open files error indicator
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == hard:  # We're at the limit
            print("Detected potential file descriptor pressure, cleaning temp directories...")
            temp_dir = tempfile.gettempdir()
            for name in os.listdir(temp_dir):
                if name.startswith(('tmp', 'wandb-')):
                    try:
                        path = os.path.join(temp_dir, name)
                        if os.path.isdir(path) and not os.listdir(path):  # Only clean empty dirs
                            print(f"Removing empty temp directory: {path}")
                            os.rmdir(path)
                    except (PermissionError, OSError):
                        pass
    except Exception:
        pass


def worker_init_fn(worker_id):
    """Initialize worker process."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)


def mp_cleanup():
    """Ensure proper cleanup of multiprocessing resources."""
    if hasattr(mp, 'current_process') and mp.current_process().name == 'MainProcess':
        # Only clean up from the main process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Try to clean up any lingering semaphores
        try:
            # Force the resource tracker to clean up
            from multiprocessing.resource_tracker import _resource_tracker
            _resource_tracker._check_trash()
        except Exception:
            pass


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train ADNI classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Increase file descriptor limit
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
        print(f"Increased file descriptor limit to 65536 (was {soft})")
    except Exception as e:
        print(f"Failed to increase file descriptor limit: {e}")

    # Load configuration
    config = Config.from_yaml(args.config)

    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Set seed
    set_seed(config.training.seed)

    # Set torch multiprocessing start method to 'spawn'
    if config.data.multiprocessing_context == "spawn":
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Method already set

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
        # Reduce wandb logging frequency for lower file operations
        os.environ["WANDB_LOG_INTERVAL"] = "60"  # Log every 60 seconds instead of default

        try:
            wandb_run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                name=config.wandb.run_name,
                config=config.to_dict()
            )
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Continuing without wandb logging...")
            wandb_run = None

    print("\n" + "="*80)
    print(f"Training config: {config.to_dict()}")
    print("\n" + "="*80)

    # Create transforms from config
    train_transform = get_transforms_from_config(
        config=config.data,
        mode="train"
    )

    val_transform = get_transforms_from_config(
        config=config.data,
        mode="val"
    )

    # Get dataset type from config (default to smartcache for backwards compatibility)
    dataset_type = getattr(config.data, "dataset_type", "cache")

    # Create datasets using the factory function
    train_dataset = create_adni_dataset(
        dataset_type=dataset_type,
        csv_path=config.data.train_csv_path,
        img_dir=config.data.img_dir,
        transform=train_transform,
        cache_rate=config.data.cache_rate,
        num_workers=config.data.cache_num_workers,
        cache_dir=config.data.cache_dir,
        classification_mode=config.data.classification_mode,
        mci_subtype_filter=config.data.mci_subtype_filter,
    )
    val_dataset = create_adni_dataset(
        dataset_type=dataset_type,
        csv_path=config.data.val_csv_path,
        img_dir=config.data.img_dir,
        transform=val_transform,
        cache_rate=config.data.cache_rate,
        num_workers=config.data.cache_num_workers,
        cache_dir=config.data.cache_dir,
        classification_mode=config.data.classification_mode,
        mci_subtype_filter=config.data.mci_subtype_filter,
    )

    # Add this code to examine class distribution
    labels = [sample['label'] for sample in train_dataset.base.data_list]

    # Create data loaders with optimized multiprocessing settings
    # Using proper worker init and cleanup to prevent semaphore leaks
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.training.num_workers > 0 else None,
        multiprocessing_context=config.data.multiprocessing_context if config.training.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.training.num_workers > 0 else None,
        multiprocessing_context=config.data.multiprocessing_context if config.training.num_workers > 0 else None,
        worker_init_fn=worker_init_fn
    )
    # Create model
    model_kwargs = {
        "pretrained_checkpoint": config.model.pretrained_checkpoint,
    }

    # Set num_classes based on classification_mode if not explicitly set in config
    if config.data.classification_mode == "CN_AD":
        model_kwargs["num_classes"] = 2
        print(f"Setting num_classes=2 for classification_mode={config.data.classification_mode}")
    else:
        model_kwargs["num_classes"] = config.model.num_classes
        print(f"Using num_classes={config.model.num_classes} from model config")

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
            "resize_size": config.data.resize_size,
            "classification_mode": config.data.classification_mode
        }

    # Add model-specific parameters for pretrained CNN
    elif config.model.name in ["rosanna_cnn", "pretrained_cnn"]:
        # Pass data configuration for classification mode and resize_size
        model_kwargs["data"] = {
            "classification_mode": config.data.classification_mode,
            "resize_size": config.data.resize_size
        }

        # Add pretrained CNN specific parameters
        if hasattr(config.model, 'freeze_encoder'):
            model_kwargs["freeze_encoder"] = config.model.freeze_encoder
        if hasattr(config.model, 'dropout'):
            model_kwargs["dropout"] = config.model.dropout
        if hasattr(config.model, 'input_channels'):
            model_kwargs["input_channels"] = config.model.input_channels

    model = ModelFactory.create_model(config.model.name, **model_kwargs)
    model = model.to(device)
    print("Model created!")

    # Initialize variables for training history
    start_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    class_weights = None

    # Check if we need to resume training from a checkpoint
    # All models handle pretrained_checkpoint through the model factory during creation
    # This section is only for resuming training from an existing training checkpoint
    # We distinguish between pretrained weights and training checkpoints by checking the checkpoint content
    if (
        hasattr(config.model, 'pretrained_checkpoint')
        and config.model.pretrained_checkpoint
        and os.path.isfile(config.model.pretrained_checkpoint)
    ):
        # Load the checkpoint to check if it's a training checkpoint or pretrained weights
        try:
            checkpoint_preview = torch.load(config.model.pretrained_checkpoint, map_location='cpu')

            # Check if this looks like a training checkpoint (has training history)
            is_training_checkpoint = (
                isinstance(checkpoint_preview, dict) and
                'epoch' in checkpoint_preview and
                checkpoint_preview.get('epoch', 0) > 0 and
                ('train_losses' in checkpoint_preview or 'val_losses' in checkpoint_preview)
            )

            if is_training_checkpoint:
                print(f"Detected training checkpoint, resuming training from: {config.model.pretrained_checkpoint}")
                try:
                    (
                        model, optimizer, scheduler, scaler, start_epoch,
                        train_losses, val_losses, train_accs, val_accs,
                        best_val_acc, loaded_weights
                    ) = load_checkpoint(config.model.pretrained_checkpoint, model)
                    # We need to increment the epoch as start_epoch is the last completed epoch
                    start_epoch += 1
                    print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.2f}%")

                    if loaded_weights is not None and config.training.use_class_weights:
                        if loaded_weights.size(0) == model_kwargs["num_classes"]:
                            class_weights = loaded_weights.to(device)
                            print(f"Using class weights from checkpoint: {class_weights}")
                        else:
                            print(f"Warning: Loaded class weights have shape {loaded_weights.size(0)} "
                                  f"but model has {model_kwargs['num_classes']} classes.")
                            print("Will recompute class weights based on current dataset.")
                            class_weights = None
                except Exception as e:
                    print(f"Error loading training checkpoint: {e}")
                    print("Starting training from scratch...")
                    start_epoch = 0
            else:
                print("Detected pretrained weights checkpoint (handled during model creation)")

        except Exception as e:
            print(f"Could not preview checkpoint {config.model.pretrained_checkpoint}: {e}")
            print("Assuming it's pretrained weights (handled during model creation)")

    # Log pretrained checkpoint usage (all models handle this uniformly now)
    if hasattr(config.model, 'pretrained_checkpoint') and config.model.pretrained_checkpoint:
        print(f"Model initialized with checkpoint from: {config.model.pretrained_checkpoint}")

    # Create optimizer
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

    # Create loss function with class weights if enabled
    num_classes = model_kwargs["num_classes"]
    class_weights = None

    if config.training.use_class_weights:
        # Verify if the dataset labels match our expected number of classes
        max_label = max(labels) if labels else 0
        if max_label >= num_classes:
            print(f"Warning: Dataset contains labels up to {max_label} "
                  f"but model only has {num_classes} output classes.")
            print("This may indicate a mismatch between dataset classification_mode and model num_classes.")

        # Compute class weights directly from the labels provided by the dataset
        # (Dataset factory should have already handled the label mapping for CN_AD mode)
        weights = compute_class_weights(
            labels=labels,
            num_classes=num_classes,
            weight_type=config.training.class_weight_type,
            manual_weights=config.training.manual_class_weights
        )

        class_weights = weights.to(device)
        print(f"Using class weights: {class_weights}")

        # Log class weights to wandb if enabled
        if wandb_run is not None:
            wandb_run.config.update({"class_weights": class_weights.cpu().numpy().tolist()})

    # Create the appropriate loss function based on configuration
    criterion = create_loss_function(
        loss_type=config.training.loss_type,
        num_classes=num_classes,
        class_weights=class_weights,
        focal_alpha=config.training.focal_alpha,
        focal_gamma=config.training.focal_gamma,
        device=device
    )

    # Log focal loss parameters to wandb if enabled
    if wandb_run is not None and config.training.loss_type.lower() == "focal":
        wandb_run.config.update({
            "loss_type": config.training.loss_type,
            "focal_alpha": config.training.focal_alpha,
            "focal_gamma": config.training.focal_gamma
        })

    # Visualize training samples if requested
    if config.training.visualize:
        print("Visualizing training samples...")
        visualize_batch(train_loader, num_samples=4,
                        save_path=os.path.join(config.training.output_dir, "train_samples.png"))

    try:
        # Training loop
        for epoch in range(start_epoch, config.training.num_epochs):
            print(f"Epoch {epoch + 1}/{config.training.num_epochs} " + "="*80)

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

            print(f"\tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            # Log training metrics to wandb every epoch
            if wandb_run is not None:
                wandb_log = {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/lr": current_lr,
                }
                wandb_run.log(wandb_log, step=epoch + 1)

            # Check if validation should be run this epoch
            should_validate = (epoch + 1) % config.training.val_epoch_freq == 0 \
                              or (epoch + 1) == config.training.num_epochs

            if should_validate:
                # Validate
                val_loss, val_acc, true_labels, predicted_labels = validate(
                    model, val_loader, criterion, device, use_mixed_precision=use_mixed_precision
                )
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                print(f"\tVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # Generate and log confusion matrix less frequently to reduce file operations
                should_generate_cm = (epoch + 1) % 5 == 0 or (epoch + 1) == config.training.num_epochs

                if should_generate_cm:
                    print(f"\tGenerating confusion matrix for epoch {epoch + 1}...")
                    cm_path = os.path.join(config.training.output_dir, f"confusion_matrix_epoch_{epoch + 1}.png")

                    # Set class names based on classification mode
                    if config.data.classification_mode == "CN_AD":
                        class_names = ["CN", "AD"]
                    else:
                        class_names = ["CN", "MCI", "AD"]

                    cm_fig = plot_confusion_matrix(
                        y_true=true_labels,
                        y_pred=predicted_labels,
                        class_names=class_names,
                        normalize=False,
                        save_path=cm_path,
                        title=f"Confusion Matrix - Epoch {epoch + 1}"
                    )

                    # Log confusion matrix to wandb
                    if wandb_run is not None:
                        wandb_run.log({
                            "val/confusion_matrix": wandb.Image(cm_fig),
                        }, step=epoch + 1)

                    # Close the figure
                    plt.close(cm_fig)
                else:
                    print(f"\tSkipping confusion matrix generation for epoch {epoch + 1} (generated every 5 epochs)")

                # Update learning rate scheduler
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_loss)  # For ReduceLROnPlateau
                    else:
                        scheduler.step()  # For other schedulers

                # Log validation metrics to wandb
                if wandb_run is not None:
                    wandb_run.log({
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                    }, step=epoch + 1)

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
                    checkpoint_config=config.training.checkpoint,
                    class_weights=class_weights
                )

                # Log sample images to WandB with predictions (same frequency as validation)
                if wandb_run is not None:
                    print("\tLogging sample images to WandB...")
                    try:
                        log_sample_images_to_wandb(
                            model=model,
                            dataset=train_dataset,
                            device=device,
                            wandb_run=wandb_run,
                            num_samples=4,
                            classification_mode=config.data.classification_mode
                        )
                    except Exception as e:
                        print(f"Error logging sample images to WandB: {e}")

                # Visualize predictions less frequently to reduce file operations
                if config.training.visualize and ((epoch + 1) % 10 == 0 or (epoch + 1) == config.training.num_epochs):
                    print("\tVisualizing predictions...")
                    try:
                        visualize_predictions(
                            model, val_loader, device, num_samples=4,
                            save_path=os.path.join(config.training.output_dir, f"predictions_epoch_{epoch + 1}.png")
                        )
                    except Exception as e:
                        print(f"Error visualizing predictions: {e}")

                    # Only clean up after potentially memory-intensive operations
                    torch.cuda.empty_cache()
            else:
                print(f"\tSkipping validation for epoch {epoch + 1} "
                      f"(validation frequency: every {config.training.val_epoch_freq} epochs)")

                # For non-plateau schedulers, we need to step even without validation
                if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()  # Step schedulers that don't depend on validation metrics

        # Plot training history
        plot_training_history(
            train_losses, val_losses, train_accs, val_accs,
            save_path=os.path.join(config.training.output_dir, "training_history.png")
        )

        # Close wandb run
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as e:
                print(f"Error closing wandb run: {e}")

    except Exception as e:
        print(f"Error during training: {e}")
        # Make sure to clean up wandb resources even on error
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        raise
    finally:
        # Final cleanup
        cleanup_resources()

        # Explicit DataLoader cleanup to prevent semaphore leaks
        # Delete DataLoaders to release worker processes
        try:
            # Delete loaders to release worker processes
            del train_loader, val_loader

            # Force GC to clean up DataLoader resources
            gc.collect()
            torch.cuda.empty_cache()

            # Sleep briefly to allow resource tracker to clean up
            time.sleep(0.5)

            # Force multiprocessing cleanup
            if hasattr(mp, '_cleanup'):
                mp._cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
