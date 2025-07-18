"""Visualization utilities for ADNI classification."""

import os
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


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
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

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
        axes[i, 0].imshow(img[mid_z, :, :], cmap="gray")
        axes[i, 0].set_title(f"Sample {i + 1}, Label: {label_name} ({label_idx}), Z-slice {mid_z}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img[:, mid_y, :], cmap="gray")
        axes[i, 1].set_title(f"Y-slice {mid_y}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img[:, :, mid_x], cmap="gray")
        axes[i, 2].set_title(f"X-slice {mid_x}")
        axes[i, 2].axis("off")

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
    save_path: Optional[str] = None,
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
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

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
        axes[i, 0].imshow(img[mid_z, :, :], cmap="gray")
        axes[i, 0].set_title(f"Sample {i + 1}, True: {true_label_name}, Pred: {pred_label_name}, Z-slice {mid_z}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img[:, mid_y, :], cmap="gray")
        axes[i, 1].set_title(f"Y-slice {mid_y}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img[:, :, mid_x], cmap="gray")
        axes[i, 2].set_title(f"X-slice {mid_x}")
        axes[i, 2].axis("off")

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
    save_path: Optional[str] = None,
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
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accs, label="Train Accuracy")
    ax2.plot(val_accs, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot (if None, display instead)
        title: Title of the plot

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ["CN", "MCI", "AD"]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use seaborn for better visualization
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    # Set labels and title
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()

    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"\tSaved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()

    return fig


def log_sample_images_to_wandb(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    wandb_run: Any,
    num_samples: int = 4,
    classification_mode: str = "CN_MCI_AD",
) -> None:
    """Log sample images with predictions to WandB.

    Args:
        model: Trained model
        dataset: Dataset to get images from
        device: Device to run the model on
        wandb_run: WandB run object
        num_samples: Number of samples to log
        classification_mode: Classification mode to determine class names
    """
    if wandb_run is None:
        return

    model.eval()

    # Create a simple DataLoader without multiprocessing to avoid conflicts
    from torch.utils.data import DataLoader

    simple_loader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=True,
        num_workers=0,  # No multiprocessing to avoid conflicts
        pin_memory=False,
    )

    # Get a batch of data
    batch = next(iter(simple_loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    # Map label indices to class names based on classification mode
    if classification_mode == "CN_AD":
        label_names = {0: "CN", 1: "AD"}
    else:
        label_names = {0: "CN", 1: "MCI", 2: "AD"}

    # Prepare images for WandB logging
    wandb_images = []

    for i in range(min(num_samples, len(images))):
        # Get the image, true label, and predicted label
        img = images[i, 0].cpu().numpy()  # Remove channel dimension
        true_label_idx = labels[i].item()
        pred_label_idx = predicted[i].item()
        pred_prob = probabilities[i, pred_label_idx].item()

        true_label_name = label_names.get(true_label_idx, f"Unknown ({true_label_idx})")
        pred_label_name = label_names.get(pred_label_idx, f"Unknown ({pred_label_idx})")

        # Get middle slices in each dimension
        mid_z = img.shape[0] // 2
        mid_y = img.shape[1] // 2
        mid_x = img.shape[2] // 2

        # Create a figure with 3 subplots for the 3 different slice views
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Plot the slices
        axes[0].imshow(img[mid_z, :, :], cmap="gray")
        axes[0].set_title(f"Z-slice {mid_z}")
        axes[0].axis("off")

        axes[1].imshow(img[:, mid_y, :], cmap="gray")
        axes[1].set_title(f"Y-slice {mid_y}")
        axes[1].axis("off")

        axes[2].imshow(img[:, :, mid_x], cmap="gray")
        axes[2].set_title(f"X-slice {mid_x}")
        axes[2].axis("off")

        plt.suptitle(
            f"Sample {i + 1} - True: {true_label_name}, Pred: {pred_label_name} ({pred_prob:.3f})", fontsize=14, y=1.02
        )
        plt.tight_layout()

        # Convert figure to image format for WandB
        import wandb

        wandb_images.append(
            wandb.Image(
                fig, caption=f"Sample {i + 1}: True={true_label_name}, Pred={pred_label_name}, Prob={pred_prob:.3f}"
            )
        )

        plt.close(fig)

        # Log images to WandB
    wandb_run.log({"sample_images": wandb_images})

    # Clean up the temporary DataLoader
    del simple_loader

    print(f"Logged {len(wandb_images)} sample images to WandB")
