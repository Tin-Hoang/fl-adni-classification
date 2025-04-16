"""Visualization utilities for ADNI classification."""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional

from adni_classification.datasets.adni_dataset import ADNIDataset, get_transforms


def visualize_batch(dataloader: DataLoader, num_samples: int = 5, save_path: Optional[str] = None) -> None:
    """Visualize a batch of images from the dataloader.

    Args:
        dataloader: DataLoader to get images from
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization (if None, display instead)
    """
    # Get a batch of data
    batch = next(iter(dataloader))
    images = batch["image"]
    labels = batch["label"]

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    # Map label indices to class names
    label_names = {0: "CN", 1: "MCI", 2: "AD"}

    # Plot each sample
    for i in range(min(num_samples, len(images))):
        # Get the image and label
        img = images[i, 0].numpy()  # Remove channel dimension
        label_idx = labels[i].item()
        label_name = label_names.get(label_idx, f"Unknown ({label_idx})")

        # Get middle slices in each dimension
        mid_z = img.shape[0] // 2
        mid_y = img.shape[1] // 2
        mid_x = img.shape[2] // 2

        # Plot the slices
        axes[i, 0].imshow(img[mid_z, :, :], cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}, Label: {label_name} ({label_idx}), Z-slice {mid_z}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img[:, mid_y, :], cmap='gray')
        axes[i, 1].set_title(f'Y-slice {mid_y}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img[:, :, mid_x], cmap='gray')
        axes[i, 2].set_title(f'X-slice {mid_x}')
        axes[i, 2].axis('off')

    plt.tight_layout()

    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 5,
    save_path: Optional[str] = None
) -> None:
    """Visualize model predictions on a batch of images.

    Args:
        model: Trained model
        dataloader: DataLoader to get images from
        device: Device to run the model on
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization (if None, display instead)
    """
    model.eval()

    # Get a batch of data
    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Map label indices to class names
    label_names = {0: "CN", 1: "MCI", 2: "AD"}

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    # Plot each sample
    for i in range(min(num_samples, len(images))):
        # Get the image, true label, and predicted label
        img = images[i, 0].cpu().numpy()  # Remove channel dimension
        true_label_idx = labels[i].item()
        pred_label_idx = predicted[i].item()

        true_label_name = label_names.get(true_label_idx, f"Unknown ({true_label_idx})")
        pred_label_name = label_names.get(pred_label_idx, f"Unknown ({pred_label_idx})")

        # Get middle slices in each dimension
        mid_z = img.shape[0] // 2
        mid_y = img.shape[1] // 2
        mid_x = img.shape[2] // 2

        # Plot the slices
        axes[i, 0].imshow(img[mid_z, :, :], cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}, True: {true_label_name}, Pred: {pred_label_name}, Z-slice {mid_z}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img[:, mid_y, :], cmap='gray')
        axes[i, 1].set_title(f'Y-slice {mid_y}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img[:, :, mid_x], cmap='gray')
        axes[i, 2].set_title(f'X-slice {mid_x}')
        axes[i, 2].axis('off')

    plt.tight_layout()

    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plot training history.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot (if None, display instead)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()
