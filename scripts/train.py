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
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import atexit
import gc
import signal
import tempfile

from adni_classification.models.model_factory import ModelFactory
from adni_classification.datasets.adni_dataset import ADNIDataset, get_transforms
from adni_classification.utils.visualization import (
    visualize_batch,
    visualize_predictions,
    plot_training_history,
    plot_confusion_matrix
)
from adni_classification.config.config import Config


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
        class_weights = [np.sqrt(total_samples / (num_classes * count)) if count > 0 else 1.0 for count in sorted_counts]
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
    try:
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

    except OSError as e:
        if "Too many open files" in str(e):
            print(f"OSError (Too many open files): {e}")
            print("Trying to recover by cleaning up resources...")
            # Force cleanup only when necessary
            gc.collect()
            torch.cuda.empty_cache()
            # Sleep to let files close
            import time
            time.sleep(2)

        if batch_idx == 0:
            # If we failed at the very first batch, re-raise to avoid silent failures
            raise
        # If we've processed at least some batches, we'll continue with what we have
        print(f"Processed {batch_idx} batches before error. Continuing with partial epoch.")
        if total == 0:
            # If no samples were processed successfully, we can't calculate metrics
            return 0.0, 0.0
    except RuntimeError as e:
        print(f"RuntimeError in DataLoader: {e}")
        if "CUDA" in str(e):
            print("CUDA error detected. Trying to recover...")
            torch.cuda.empty_cache()
        if batch_idx == 0:
            # If we failed at the very first batch, re-raise to avoid silent failures
            raise
        # If we've processed at least some batches, we'll continue with what we have
        print(f"Processed {batch_idx} batches before error. Continuing with partial epoch.")
        if total == 0:
            # If no samples were processed successfully, we can't calculate metrics
            return 0.0, 0.0

    # Calculate final metrics based on what was successfully processed
    avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0
    avg_accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Only clean up at the end of the epoch
    torch.cuda.empty_cache()

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
        try:
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

        except OSError as e:
            if "Too many open files" in str(e):
                print(f"OSError (Too many open files): {e}")
                print("Trying to recover by cleaning up resources...")
                # Force cleanup only when necessary
                gc.collect()
                torch.cuda.empty_cache()
                # Sleep to let files close
                import time
                time.sleep(2)

            if batch_idx == 0:
                # If we failed at the very first batch, re-raise to avoid silent failures
                raise
            # If we've processed at least some batches, we'll continue with what we have
            print(f"Processed {batch_idx} batches before error. Continuing with partial validation.")
            if total == 0:
                # If no samples were processed successfully, we can't calculate metrics
                return 0.0, 0.0, np.array([]), np.array([])
        except RuntimeError as e:
            print(f"RuntimeError in validation DataLoader: {e}")
            if "CUDA" in str(e):
                print("CUDA error detected. Trying to recover...")
                torch.cuda.empty_cache()
            if batch_idx == 0:
                # If we failed at the very first batch, re-raise to avoid silent failures
                raise
            # If we've processed at least some batches, we'll continue with what we have
            print(f"Processed {batch_idx} batches before error. Continuing with partial validation.")
            if total == 0:
                # If no samples were processed successfully, we can't calculate metrics
                return 0.0, 0.0, np.array([]), np.array([])

    # Calculate metrics based on what was successfully processed
    avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0
    avg_accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Only clean up at the end of validation
    torch.cuda.empty_cache()

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
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Optional[GradScaler], int, List[float], List[float], List[float], List[float], float, Optional[torch.Tensor]]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into (optional)
        scheduler: The learning rate scheduler to load state into (optional)
        scaler: The GradScaler to load state into (optional)

    Returns:
        Tuple of (model, optimizer, scheduler, scaler, start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc, class_weights)
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

    # Get class weights if they exist
    class_weights = checkpoint.get('class_weights', None)

    return model, optimizer, scheduler, scaler, start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc, class_weights


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


def cleanup_resources():
    """Clean up resources to prevent file descriptor leaks."""
    print("Cleaning up resources...")

    # Only perform critical cleanup operations
    # 1. Empty CUDA cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
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
    except:
        pass


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train ADNI classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Increase file descriptor limit if possible
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to increase the limit to the hard limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Increased file descriptor limit from {soft} to {hard}")
    except (ImportError, ValueError, resource.error):
        print("Could not increase file descriptor limit")

    # Set torch multiprocessing start method to 'spawn'
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Method already set

    # Set globals to properly clean up multiprocessing resources
    # These settings help prevent semaphore leaks
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Register a proper cleanup handler for multiprocessing resources
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
            except:
                pass

    atexit.register(mp_cleanup)

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

    # Create datasets using the factory function with cache parameters from config
    train_dataset = ADNIDataset(
        csv_path=config.data.train_csv_path,
        img_dir=config.data.img_dir,
        transform=train_transform,
        cache_rate=config.data.cache_rate,
        num_workers=config.data.cache_num_workers
    )

    val_dataset = ADNIDataset(
        csv_path=config.data.val_csv_path,
        img_dir=config.data.img_dir,
        transform=val_transform,
        cache_rate=config.data.cache_rate,
        num_workers=config.data.cache_num_workers
    )

    # Add this code to examine class distribution
    labels = [sample["label"].item() for sample in train_dataset]

    # Ensure num_workers is at least 1 to enable multiprocessing
    num_workers = max(1, config.training.num_workers)

    # Create data loaders with optimized multiprocessing settings
    # Using proper worker init and cleanup to prevent semaphore leaks
    def worker_init_fn(worker_id):
        # Set different seeds for different workers for better randomization
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        # Make sure each worker has its own CUDA context to avoid conflicts
        torch.cuda.set_device(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Re-enable pin memory for performance
        persistent_workers=True,  # Keep workers alive between batches
        prefetch_factor=2,  # Prefetch 2 batches per worker
        multiprocessing_context='spawn',  # Use spawn for better compatibility
        worker_init_fn=worker_init_fn,  # Initialize workers properly
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='spawn',
        worker_init_fn=worker_init_fn,  # Initialize workers properly
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

    # Initialize variables for training history
    start_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    class_weights = None

    # Check if we need to resume training from a checkpoint
    if config.model.pretrained_checkpoint:
        # Check if the weights path is a checkpoint file
        if os.path.isfile(config.model.pretrained_checkpoint):
            print(f"Loading checkpoint from: {config.model.pretrained_checkpoint}")
            model, optimizer, scheduler, scaler, start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc, loaded_weights = load_checkpoint(
                config.model.pretrained_checkpoint, model
            )
            # We need to increment the epoch as start_epoch is the last completed epoch
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.2f}%")

            # If class weights were in the checkpoint and we're using class weights, use them
            if loaded_weights is not None and config.training.use_class_weights:
                class_weights = loaded_weights.to(device)
                print(f"Using class weights from checkpoint: {class_weights}")
        else:
            print(f"No checkpoint found at: {config.model.pretrained_checkpoint}")
            # If it's not a checkpoint file, it might be just a state dict
            if os.path.isfile(config.model.pretrained_checkpoint):
                print(f"Loading model state dict from: {config.model.pretrained_checkpoint}")
                state_dict = torch.load(config.model.pretrained_checkpoint, map_location=device)
                model.load_state_dict(state_dict)
                print(f"Loaded model state dict from: {config.model.pretrained_checkpoint}")

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

    # If we resumed from a checkpoint, update optimizer and scheduler with loaded states
    if config.model.pretrained_checkpoint and os.path.isfile(config.model.pretrained_checkpoint):
        _, optimizer, scheduler, scaler, _, _, _, _, _, _, _ = load_checkpoint(
            config.model.pretrained_checkpoint, model, optimizer, scheduler, scaler
        )

    # Create loss function with class weights if enabled
    if config.training.use_class_weights:
        # If we didn't get class weights from a checkpoint, compute them
        if class_weights is None:
            class_weights = compute_class_weights(
                labels=labels,
                num_classes=config.model.num_classes,
                weight_type=config.training.class_weight_type,
                manual_weights=config.training.manual_class_weights
            )
            class_weights = class_weights.to(device)

        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Log class weights to wandb if enabled
        if wandb_run is not None:
            wandb_run.config.update({"class_weights": class_weights.cpu().numpy().tolist()})
    else:
        criterion = nn.CrossEntropyLoss()
        class_weights = None

    # Visualize training samples if requested
    if config.training.visualize:
        print("Visualizing training samples...")
        visualize_batch(train_loader, num_samples=4, save_path=os.path.join(config.training.output_dir, "train_samples.png"))

    # Add cleanup handlers to ensure resources are released
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_resources(), exit(0)))

    try:
        # Training loop
        for epoch in range(start_epoch, config.training.num_epochs):
            print(f"Epoch {epoch + 1}/{config.training.num_epochs} " + "="*40)

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
            should_validate = (epoch + 1) % config.training.val_epoch_freq == 0 or (epoch + 1) == config.training.num_epochs

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
                    cm_fig = plot_confusion_matrix(
                        y_true=true_labels,
                        y_pred=predicted_labels,
                        class_names=["CN", "MCI", "AD"],
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
                print(f"\tSkipping validation for epoch {epoch + 1} (validation frequency: every {config.training.val_epoch_freq} epochs)")

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
            except:
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
            import time
            time.sleep(0.5)

            # Force multiprocessing cleanup
            if hasattr(mp, '_cleanup'):
                mp._cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
