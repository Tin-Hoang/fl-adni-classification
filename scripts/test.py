"""Evaluation script for ADNI classification models."""

import argparse
import gc
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


from adni_classification.datasets.dataset_factory import create_adni_dataset
from adni_classification.models.model_factory import ModelFactory
from adni_classification.utils.visualization import plot_confusion_matrix, visualize_predictions


def detect_model_name_from_checkpoint(checkpoint_path: str) -> str:
    """Detect model name from checkpoint path or filename.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Detected model name
    """
    # Extract filename and directory
    checkpoint_filename = os.path.basename(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Common model name patterns in filenames
    filename_lower = checkpoint_filename.lower()

    if "rosanna" in filename_lower or "pretrained_cnn" in filename_lower:
        return "rosanna_cnn"
    elif "securefed" in filename_lower:
        return "securefed_cnn"
    elif "resnet" in filename_lower:
        return "resnet3d"
    elif "densenet" in filename_lower:
        return "densenet3d"
    elif "simple" in filename_lower:
        return "simple3dcnn"

    # Check directory name for patterns
    dir_name_lower = checkpoint_dir.lower()
    if "rosanna" in dir_name_lower:
        return "rosanna_cnn"
    elif "securefed" in dir_name_lower:
        return "securefed_cnn"
    elif "resnet" in dir_name_lower:
        return "resnet3d"
    elif "densenet" in dir_name_lower:
        return "densenet3d"

    # Default fallback
    print(f"Warning: Could not auto-detect model name from {checkpoint_path}")
    print("Please specify --model_name explicitly")
    print("Defaulting to 'resnet3d'")
    return "resnet3d"


def create_automatic_output_dir(checkpoint_path: str, base_dir: str = "evaluation_results") -> str:
    """Create automatic output directory based on checkpoint path and timestamp.

    Args:
        checkpoint_path: Path to the checkpoint file
        base_dir: Base directory for outputs

    Returns:
        Generated output directory path

    Example:
        checkpoint_path = "outputs_centralized/rosanna-3T_P1-1220images-2classes-scratch-seed101_20250717_034328/rosanna_cnn_checkpoint_best.pth"
        Returns: "evaluation_results/rosanna-3T_P1-1220images-2classes-scratch-seed101_20250717_034328_best_20250118_143025"
    """
    # Get the parent directory name (the training output folder)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    parent_dir_name = os.path.basename(checkpoint_dir)

    # Extract checkpoint type from filename
    checkpoint_filename = os.path.basename(checkpoint_path)
    checkpoint_type = "unknown"

    if "best" in checkpoint_filename.lower():
        checkpoint_type = "best"
    elif "latest" in checkpoint_filename.lower():
        checkpoint_type = "latest"
    elif "epoch" in checkpoint_filename.lower():
        # Extract epoch number if present
        import re
        epoch_match = re.search(r'epoch_(\d+)', checkpoint_filename.lower())
        if epoch_match:
            checkpoint_type = f"epoch{epoch_match.group(1)}"
        else:
            checkpoint_type = "epoch"
    elif "checkpoint" in checkpoint_filename.lower():
        checkpoint_type = "checkpoint"

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the output directory name
    output_dir = os.path.join(base_dir, f"{parent_dir_name}_{checkpoint_type}_{timestamp}")

    return output_dir


def load_model_from_checkpoint(
    checkpoint_path: str,
    args: Any,
    device: torch.device
) -> nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        args: Command line arguments object
        device: Device to load the model on

    Returns:
        Loaded model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Create model with the same configuration
    model_kwargs = {}

    # Set num_classes based on classification_mode
    if args.classification_mode == "CN_AD":
        model_kwargs["num_classes"] = 2
        print(f"Setting num_classes=2 for classification_mode={args.classification_mode}")
    else:
        model_kwargs["num_classes"] = 3  # Default for CN_MCI_AD
        print(f"Setting num_classes=3 for classification_mode={args.classification_mode}")

    # Add model-specific parameters with defaults
    if args.model_name == "resnet3d":
        model_kwargs["model_depth"] = 18  # Default depth
    elif args.model_name == "densenet3d":
        # Use DenseNet defaults if not specified
        pass  # ModelFactory will use defaults

    # Handle SecureFedCNN configuration
    if args.model_name == "securefed_cnn":
        model_kwargs["data"] = {
            "resize_size": args.resize_size,
            "classification_mode": args.classification_mode
        }

    # Handle RosannaCNN configuration
    elif args.model_name in ["rosanna_cnn", "pretrained_cnn"]:
        model_kwargs["data"] = {
            "resize_size": args.resize_size,
            "classification_mode": args.classification_mode
        }

        # For RosannaCNN, add parameters directly with defaults
        model_kwargs["freeze_encoder"] = False  # Default value
        model_kwargs["dropout"] = 0.0  # Default value
        model_kwargs["input_channels"] = 1  # Default to 1 channel for MRI data

    # Create model (ensure no pretrained_checkpoint is passed for test mode)
    print(f"Creating model with kwargs: {model_kwargs}")
    model = ModelFactory.create_model(args.model_name, **model_kwargs)
    model = model.to(device)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Training checkpoint format
                state_dict = checkpoint['model_state_dict']
                print("Loading from training checkpoint format")
            elif 'state_dict' in checkpoint:
                # Alternative checkpoint format
                state_dict = checkpoint['state_dict']
                print("Loading from state_dict checkpoint format")
            else:
                # Assume the entire dict is the state dict
                state_dict = checkpoint
                print("Loading from direct state dict format")
        else:
            # Direct state dict
            state_dict = checkpoint
            print("Loading from direct state dict")

        # Load with shape checking
        current_state_dict = model.state_dict()

        # Check for shape mismatches
        mismatched_keys = []
        for key, value in state_dict.items():
            if key in current_state_dict:
                if current_state_dict[key].shape != value.shape:
                    mismatched_keys.append(f"{key}: expected {current_state_dict[key].shape}, got {value.shape}")

        if mismatched_keys:
            print("Warning: Found shape mismatches in checkpoint:")
            for mismatch in mismatched_keys:
                print(f"  {mismatch}")
            print("Loading with strict=False to skip mismatched layers")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

        print("✓ Model loaded successfully")

        # Print additional checkpoint info if available
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"Checkpoint info: epoch {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                print(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    return model


def calculate_dataset_statistics(dataset: Any, classification_mode: str) -> Dict[str, Any]:
    """Calculate statistics for the dataset.

    Args:
        dataset: The dataset to analyze
        classification_mode: Classification mode (CN_AD or CN_MCI_AD)

    Returns:
        Dictionary containing dataset statistics
    """
    print("Calculating dataset statistics...")

    # Get all labels
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if isinstance(sample, dict):
            label = sample['label']
            if hasattr(label, 'item'):
                label = label.item()
            labels.append(label)
        else:
            # Handle tuple format (image, label)
            labels.append(sample[1])

    labels = np.array(labels)

    # Class mapping
    if classification_mode == "CN_AD":
        class_names = ["CN", "AD"]
        num_classes = 2
    else:
        class_names = ["CN", "MCI", "AD"]
        num_classes = 3

    # Calculate statistics
    unique_labels, counts = np.unique(labels, return_counts=True)

    stats = {
        "total_samples": len(labels),
        "num_classes": num_classes,
        "class_names": class_names,
        "class_distribution": {class_names[i]: int(counts[np.where(unique_labels == i)[0][0]])
                              if i in unique_labels else 0 for i in range(num_classes)},
        "class_percentages": {},
        "min_class_size": int(np.min(counts)),
        "max_class_size": int(np.max(counts)),
        "class_imbalance_ratio": float(np.max(counts) / np.min(counts)),
    }

    # Calculate percentages
    total = stats["total_samples"]
    for class_name, count in stats["class_distribution"].items():
        stats["class_percentages"][class_name] = round(100.0 * count / total, 2)

    return stats


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    classification_mode: str,
    dataset: Any = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, float, List[str]]:
    """Evaluate the model on test data.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        classification_mode: Classification mode
        dataset: Dataset object to extract image paths from

    Returns:
        Tuple of (metrics_dict, true_labels, predicted_labels, predicted_probs, inference_time, image_paths)
    """
    print("Running model evaluation...")

    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_image_paths = []
    total_inference_time = 0.0

        # Progress bar
    pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    batch_index = 0

    with torch.no_grad():
        for batch in pbar:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

            batch_size = images.size(0)

            # Extract image paths from dataset if available
            if dataset is not None:
                # Try different ways to access dataset data
                data_list = None
                if hasattr(dataset, 'data_list'):
                    data_list = dataset.data_list
                elif hasattr(dataset, 'base') and hasattr(dataset.base, 'data_list'):
                    # For datasets that wrap a base dataset
                    data_list = dataset.base.data_list
                elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'data_list'):
                    # For other wrapped datasets
                    data_list = dataset.dataset.data_list

                if data_list is not None:
                    # Calculate the indices for this batch (handle shuffling correctly)
                    start_idx = batch_index * test_loader.batch_size
                    batch_image_paths = []
                    for i in range(batch_size):
                        idx = start_idx + i
                        if idx < len(data_list):
                            # Handle different data structures
                            item = data_list[idx]
                            if isinstance(item, dict) and "image" in item:
                                image_path = item["image"]
                            elif hasattr(item, 'get'):
                                image_path = item.get("image", f"image_{idx}")
                            else:
                                image_path = f"image_{idx}"
                            batch_image_paths.append(str(image_path))
                        else:
                            batch_image_paths.append(f"image_{idx}")
                else:
                    # Fallback: generate placeholder names
                    batch_image_paths = [f"image_{len(all_image_paths) + i}" for i in range(batch_size)]
            else:
                # Fallback: generate placeholder names
                batch_image_paths = [f"image_{len(all_image_paths) + i}" for i in range(batch_size)]

            batch_index += 1

            # Time inference
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # Collect image paths
            if isinstance(batch_image_paths, (list, tuple)):
                all_image_paths.extend(batch_image_paths)
            else:
                # Handle single path or tensor
                if hasattr(batch_image_paths, 'tolist'):
                    all_image_paths.extend(batch_image_paths.tolist())
                else:
                    all_image_paths.append(str(batch_image_paths))

    # Convert to numpy arrays
    true_labels = np.array(all_labels)
    predicted_labels = np.array(all_predictions)
    predicted_probs = np.array(all_probabilities)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Class names
    if classification_mode == "CN_AD":
        class_names = ["CN", "AD"]
        num_classes = 2
    else:
        class_names = ["CN", "MCI", "AD"]
        num_classes = 3

    # Precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )

    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0
    )

    # AUC calculation
    auc_scores = {}
    if num_classes == 2:
        # Binary classification - single AUC
        if len(np.unique(true_labels)) == 2:  # Ensure both classes are present
            auc_scores["binary"] = roc_auc_score(true_labels, predicted_probs[:, 1])
    else:
        # Multi-class classification - AUC per class (one-vs-rest)
        try:
            auc_per_class = roc_auc_score(true_labels, predicted_probs, multi_class='ovr', average=None)
            for i, class_name in enumerate(class_names):
                if i < len(auc_per_class):
                    auc_scores[f"{class_name}_vs_rest"] = auc_per_class[i]
            # Macro average
            auc_scores["macro_average"] = roc_auc_score(true_labels, predicted_probs, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Could not calculate AUC: {e}")

    # Top-k accuracy (useful for multi-class)
    top_k_acc = {}
    if num_classes > 2:
        for k in [2, 3]:
            if k <= num_classes:
                # Get top-k predictions
                top_k_preds = np.argsort(predicted_probs, axis=1)[:, -k:]
                top_k_correct = np.array([true_labels[i] in top_k_preds[i] for i in range(len(true_labels))])
                top_k_acc[f"top_{k}"] = np.mean(top_k_correct)

    # Confidence analysis
    max_probs = np.max(predicted_probs, axis=1)
    confidence_stats = {
        "mean_confidence": float(np.mean(max_probs)),
        "std_confidence": float(np.std(max_probs)),
        "min_confidence": float(np.min(max_probs)),
        "max_confidence": float(np.max(max_probs)),
    }

    # Organize metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision_per_class": {class_names[i]: float(precision[i]) for i in range(len(precision))},
        "recall_per_class": {class_names[i]: float(recall[i]) for i in range(len(recall))},
        "f1_per_class": {class_names[i]: float(f1[i]) for i in range(len(f1))},
        "support_per_class": {class_names[i]: int(support[i]) for i in range(len(support))},
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "auc_scores": auc_scores,
        "top_k_accuracy": top_k_acc,
        "confidence_stats": confidence_stats,
        "total_inference_time": total_inference_time,
        "avg_inference_time_per_batch": total_inference_time / len(test_loader),
        "avg_inference_time_per_sample": total_inference_time / len(true_labels),
    }

    return metrics, true_labels, predicted_labels, predicted_probs, total_inference_time, all_image_paths


def plot_roc_curves(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    classification_mode: str,
    save_path: str
) -> None:
    """Plot ROC curves.

    Args:
        true_labels: True labels
        predicted_probs: Predicted probabilities
        classification_mode: Classification mode
        save_path: Path to save the plot
    """
    if classification_mode == "CN_AD":
        class_names = ["CN", "AD"]
        num_classes = 2
    else:
        class_names = ["CN", "MCI", "AD"]
        num_classes = 3

    plt.figure(figsize=(10, 8))

    if num_classes == 2:
        # Binary classification
        if len(np.unique(true_labels)) == 2:
            fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, 1])
            auc = roc_auc_score(true_labels, predicted_probs[:, 1])
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        else:
            print("Warning: Only one class present in test set, cannot plot ROC curve")
            return
    else:
        # Multi-class classification (one-vs-rest)
        try:
            for i, class_name in enumerate(class_names):
                # Create binary labels for current class vs rest
                binary_labels = (true_labels == i).astype(int)
                if len(np.unique(binary_labels)) == 2:  # Both classes present
                    fpr, tpr, _ = roc_curve(binary_labels, predicted_probs[:, i])
                    auc = roc_auc_score(binary_labels, predicted_probs[:, i])
                    plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')
        except ValueError as e:
            print(f"Could not plot ROC curves: {e}")
            return

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {save_path}")


def plot_confidence_histogram(
    predicted_probs: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    classification_mode: str,
    save_path: str
) -> None:
    """Plot confidence score histogram.

    Args:
        predicted_probs: Predicted probabilities
        predicted_labels: Predicted labels
        true_labels: True labels
        classification_mode: Classification mode
        save_path: Path to save the plot
    """
    if classification_mode == "CN_AD":
        class_names = ["CN", "AD"]
    else:
        class_names = ["CN", "MCI", "AD"]

    # Get maximum probability (confidence) for each prediction
    max_probs = np.max(predicted_probs, axis=1)

    # Separate correct and incorrect predictions
    correct_mask = predicted_labels == true_labels
    correct_confidences = max_probs[correct_mask]
    incorrect_confidences = max_probs[~correct_mask]

    plt.figure(figsize=(12, 6))

    # Plot histograms
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot confidence by class
    plt.subplot(1, 2, 2)
    for i, class_name in enumerate(class_names):
        class_mask = predicted_labels == i
        if np.any(class_mask):
            class_confidences = max_probs[class_mask]
            plt.hist(class_confidences, bins=20, alpha=0.7, label=f'{class_name}')

    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score by Predicted Class')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence histogram to {save_path}")


def save_results(
    results: Dict[str, Any],
    save_path: str
) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Results dictionary
        save_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {save_path}")


def save_predictions_csv(
    image_paths: List[str],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_probs: np.ndarray,
    classification_mode: str,
    save_path: str
) -> None:
    """Save detailed predictions to CSV file.

    Args:
        image_paths: List of image file paths
        true_labels: True labels
        predicted_labels: Predicted labels
        predicted_probs: Predicted probabilities for each class
        classification_mode: Classification mode
        save_path: Path to save the CSV file
    """
    # Class names mapping
    if classification_mode == "CN_AD":
        class_names = ["CN", "AD"]
    else:
        class_names = ["CN", "MCI", "AD"]

    # Prepare data for CSV
    csv_data = []

    for i in range(len(image_paths)):
        # Basic information
        row = {
            "image_path": image_paths[i],
            "image_filename": os.path.basename(image_paths[i]) if isinstance(image_paths[i], str) else str(image_paths[i]),
            "true_label_numeric": int(true_labels[i]),
            "true_label_name": class_names[int(true_labels[i])],
            "predicted_label_numeric": int(predicted_labels[i]),
            "predicted_label_name": class_names[int(predicted_labels[i])],
            "prediction_correct": bool(true_labels[i] == predicted_labels[i]),
            "max_confidence": float(np.max(predicted_probs[i])),
        }

        # Add probability for each class
        for j, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(predicted_probs[i][j])

        # Add confidence margin (difference between top 2 predictions)
        sorted_probs = np.sort(predicted_probs[i])[::-1]  # Sort descending
        if len(sorted_probs) >= 2:
            row["confidence_margin"] = float(sorted_probs[0] - sorted_probs[1])
        else:
            row["confidence_margin"] = float(sorted_probs[0])

        csv_data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(csv_data)

    # Sort by image filename for easier browsing
    df = df.sort_values("image_filename")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Saved detailed predictions to {save_path}")

    # Print summary statistics
    correct_predictions = df["prediction_correct"].sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print(f"Prediction Summary:")
    print(f"  Total samples: {total_predictions}")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Mean confidence: {df['max_confidence'].mean():.4f}")
    print(f"  Mean confidence margin: {df['confidence_margin'].mean():.4f}")

    # Per-class breakdown
    print(f"  Per-class accuracy:")
    for class_name in class_names:
        class_df = df[df["true_label_name"] == class_name]
        if len(class_df) > 0:
            class_acc = class_df["prediction_correct"].mean()
            print(f"    {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {len(class_df)} samples")


def print_summary(
    dataset_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    classification_mode: str
) -> None:
    """Print evaluation summary.

    Args:
        dataset_stats: Dataset statistics
        metrics: Evaluation metrics
        classification_mode: Classification mode
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {dataset_stats['total_samples']}")
    print(f"  Number of classes: {dataset_stats['num_classes']}")
    print(f"  Classification mode: {classification_mode}")
    print(f"  Class distribution:")
    for class_name, count in dataset_stats['class_distribution'].items():
        percentage = dataset_stats['class_percentages'][class_name]
        print(f"    {class_name}: {count} samples ({percentage}%)")
    print(f"  Class imbalance ratio: {dataset_stats['class_imbalance_ratio']:.2f}")

    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1-Score: {metrics['f1_weighted']:.4f}")

    print(f"\nPer-Class Metrics:")
    class_names = list(metrics['precision_per_class'].keys())
    print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 58)
    for class_name in class_names:
        precision = metrics['precision_per_class'][class_name]
        recall = metrics['recall_per_class'][class_name]
        f1 = metrics['f1_per_class'][class_name]
        support = metrics['support_per_class'][class_name]
        print(f"{class_name:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")

    # AUC scores
    if metrics['auc_scores']:
        print(f"\nAUC Scores:")
        for auc_name, auc_value in metrics['auc_scores'].items():
            print(f"  {auc_name}: {auc_value:.4f}")

    # Top-k accuracy
    if metrics['top_k_accuracy']:
        print(f"\nTop-K Accuracy:")
        for k_name, k_value in metrics['top_k_accuracy'].items():
            print(f"  {k_name}: {k_value:.4f} ({k_value*100:.2f}%)")

    # Timing information
    print(f"\nTiming Information:")
    print(f"  Total inference time: {metrics['total_inference_time']:.4f} seconds")
    print(f"  Average time per batch: {metrics['avg_inference_time_per_batch']:.4f} seconds")
    print(f"  Average time per sample: {metrics['avg_inference_time_per_sample']:.6f} seconds")

    # Confidence statistics
    conf_stats = metrics['confidence_stats']
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {conf_stats['mean_confidence']:.4f}")
    print(f"  Std confidence: {conf_stats['std_confidence']:.4f}")
    print(f"  Min confidence: {conf_stats['min_confidence']:.4f}")
    print(f"  Max confidence: {conf_stats['max_confidence']:.4f}")

    print("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate ADNI classification model")

    # Essential arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing MRI images")

    # Model and data configuration
    parser.add_argument("--model_name", type=str, default=None, help="Model architecture (auto-detect from checkpoint if not specified)")
    parser.add_argument("--classification_mode", type=str, choices=["CN_MCI_AD", "CN_AD"], default="CN_MCI_AD", help="Classification mode")
    parser.add_argument("--mci_subtype_filter", type=str, nargs="*", default=None, help="MCI subtypes to include for CN_AD mode (e.g., EMCI LMCI)")
    parser.add_argument("--resize_size", type=int, nargs=3, default=[128, 128, 128], help="Image resize dimensions [D H W]")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results (default: auto-generated)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--num_samples_viz", type=int, default=8, help="Number of samples to visualize")

    args = parser.parse_args()

    # Create output directory (auto-generate if not provided)
    if args.output_dir is None:
        args.output_dir = create_automatic_output_dir(args.checkpoint)
        print(f"Auto-generated output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect or use provided model name
    if args.model_name is None:
        args.model_name = detect_model_name_from_checkpoint(args.checkpoint)
        print(f"Auto-detected model name: {args.model_name}")
    else:
        print(f"Using specified model name: {args.model_name}")

    # Handle MCI subtype filter
    mci_subtype_filter = args.mci_subtype_filter
    if mci_subtype_filter is not None and len(mci_subtype_filter) == 0:
        mci_subtype_filter = None

    print("Using command line arguments for all configuration")

    print(f"Test CSV: {args.test_csv}")
    print(f"Images directory: {args.img_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Classification mode: {args.classification_mode}")
    print(f"Resize size: {args.resize_size}")
    if mci_subtype_filter:
        print(f"MCI subtype filter: {mci_subtype_filter}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args, device)

    # Create test transforms from args
    from adni_classification.datasets.transforms import get_transforms
    test_transform = get_transforms(
        mode="val",
        resize_size=args.resize_size,
        resize_mode="trilinear",
        use_spacing=False,
        spacing_size=(1.0, 1.0, 1.0),
        device=None
    )

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = create_adni_dataset(
        dataset_type="normal",  # Always use normal for testing (no caching)
        csv_path=args.test_csv,
        img_dir=args.img_dir,
        transform=test_transform,
        cache_rate=0.0,  # No caching for test
        num_workers=0,  # No workers for cache in test
        cache_dir="./cache",
        classification_mode=args.classification_mode,
        mci_subtype_filter=mci_subtype_filter,
    )

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test dataset created with {len(test_dataset)} samples")

    # Calculate dataset statistics
    dataset_stats = calculate_dataset_statistics(test_dataset, args.classification_mode)

    # Evaluate model
    metrics, true_labels, predicted_labels, predicted_probs, inference_time, image_paths = evaluate_model(
        model, test_loader, device, args.classification_mode, test_dataset
    )

    # Print summary
    print_summary(dataset_stats, metrics, args.classification_mode)

    # Generate confusion matrix
    if args.classification_mode == "CN_AD":
        class_names = ["CN", "AD"]
    else:
        class_names = ["CN", "MCI", "AD"]

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm_fig = plot_confusion_matrix(
        y_true=true_labels,
        y_pred=predicted_labels,
        class_names=class_names,
        normalize=False,
        save_path=cm_path,
        title="Test Set Confusion Matrix"
    )
    plt.close(cm_fig)

    # Generate normalized confusion matrix
    cm_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalized.png")
    cm_norm_fig = plot_confusion_matrix(
        y_true=true_labels,
        y_pred=predicted_labels,
        class_names=class_names,
        normalize=True,
        save_path=cm_norm_path,
        title="Test Set Confusion Matrix (Normalized)"
    )
    plt.close(cm_norm_fig)

        # Generate ROC curves
    roc_path = os.path.join(args.output_dir, "roc_curves.png")
    plot_roc_curves(true_labels, predicted_probs, args.classification_mode, roc_path)

    # Generate confidence histogram
    conf_path = os.path.join(args.output_dir, "confidence_histogram.png")
    plot_confidence_histogram(
        predicted_probs, predicted_labels, true_labels,
        args.classification_mode, conf_path
    )

    # Save detailed predictions to CSV
    predictions_csv_path = os.path.join(args.output_dir, "predictions_detailed.csv")
    save_predictions_csv(
        image_paths, true_labels, predicted_labels, predicted_probs,
        args.classification_mode, predictions_csv_path
    )

    # Visualize predictions if requested
    if args.visualize:
        print("Generating prediction visualizations...")
        num_samples_viz = min(args.num_samples_viz, args.batch_size)
        pred_viz_path = os.path.join(args.output_dir, "prediction_samples.png")
        try:
            visualize_predictions(
                model, test_loader, device,
                num_samples=num_samples_viz,
                save_path=pred_viz_path
            )
        except Exception as e:
            print(f"Error generating prediction visualizations: {e}")

    # Prepare results for saving
    results = {
        "dataset_statistics": dataset_stats,
        "evaluation_metrics": metrics,
        "configuration": {
            "model_name": args.model_name,
            "classification_mode": args.classification_mode,
            "test_csv": args.test_csv,
            "img_dir": args.img_dir,
            "checkpoint": args.checkpoint,
            "batch_size": args.batch_size,
            "resize_size": args.resize_size,
            "mci_subtype_filter": mci_subtype_filter,
            "device": str(device),
        },
        "predictions": {
            "true_labels": true_labels.tolist(),
            "predicted_labels": predicted_labels.tolist(),
            "predicted_probabilities": predicted_probs.tolist(),
        }
    }

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    save_results(results, results_path)

    # Save detailed classification report
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("ADNI Model Evaluation Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Classification Mode: {args.classification_mode}\n")
        f.write(f"Test Dataset: {args.test_csv}\n")
        f.write(f"Images Directory: {args.img_dir}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Resize Size: {args.resize_size}\n")
        if mci_subtype_filter:
            f.write(f"MCI Subtype Filter: {mci_subtype_filter}\n")
        f.write("\n")

        f.write("Dataset Statistics:\n")
        f.write("-"*20 + "\n")
        for key, value in dataset_stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Classification Report:\n")
        f.write("-"*20 + "\n")
        f.write(classification_report(true_labels, predicted_labels, target_names=class_names))
        f.write("\n")

        f.write("Additional Metrics:\n")
        f.write("-"*20 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        if metrics['auc_scores']:
            for auc_name, auc_value in metrics['auc_scores'].items():
                f.write(f"AUC {auc_name}: {auc_value:.4f}\n")
        f.write(f"Total Inference Time: {metrics['total_inference_time']:.4f} seconds\n")
        f.write(f"Average Time per Sample: {metrics['avg_inference_time_per_sample']:.6f} seconds\n")

    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")
    print(f"Generated files:")
    print(f"  - evaluation_results.json")
    print(f"  - classification_report.txt")
    print(f"  - predictions_detailed.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - confusion_matrix_normalized.png")
    print(f"  - roc_curves.png")
    print(f"  - confidence_histogram.png")
    if args.visualize:
        print(f"  - prediction_samples.png")

    # Cleanup
    del model, test_dataset, test_loader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
