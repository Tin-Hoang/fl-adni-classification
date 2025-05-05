"""adni_flwr: A Flower-based Federated Learning framework for ADNI classification."""

import os
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np

from adni_classification.models.model_factory import ModelFactory
from adni_classification.datasets.dataset_factory import create_adni_dataset, get_transforms_from_config
from adni_classification.config.config import Config
from adni_classification.utils.torch_utils import set_seed


def load_model(config: Config) -> nn.Module:
    """Load a model based on the configuration.

    Args:
        config: Model configuration

    Returns:
        The instantiated model
    """
    model_kwargs = {
        "pretrained_checkpoint": config.model.pretrained_checkpoint,
    }

    # Set num_classes based on classification_mode if not explicitly set in config
    if config.data.classification_mode == "CN_AD":
        model_kwargs["num_classes"] = 2
    else:
        model_kwargs["num_classes"] = config.model.num_classes

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

    model = ModelFactory.create_model(config.model.name, **model_kwargs)
    return model


def get_params(model: nn.Module) -> List[np.ndarray]:
    """Get model parameters as a list of NumPy arrays.

    Args:
        model: The model

    Returns:
        List of NumPy arrays representing the model parameters
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: nn.Module, params: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays.

    Args:
        model: The model
        params: List of NumPy arrays representing the model parameters
    """
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_num: int = 1,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1
) -> Tuple[float, float]:
    """Train the model for the specified number of epochs.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        epoch_num: Number of epochs to train
        mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps for gradient accumulation

    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.to(device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    for _ in range(epoch_num):
        batch_idx = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Mixed precision training
            if mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            batch_idx += 1

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_accuracy


def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = False
) -> Tuple[float, float]:
    """Evaluate the model on the test set.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for evaluation
        mixed_precision: Whether to use mixed precision

    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_accuracy


def load_data(
    config: Config,
    batch_size: int = None
) -> Tuple[DataLoader, DataLoader]:
    """Load ADNI dataset based on configuration.

    Args:
        config: Configuration dictionary
        batch_size: Optional batch size (overrides config if provided)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set the seed for reproducibility
    set_seed(config.training.seed)

    # Use provided batch size or the one from config
    if batch_size is None:
        batch_size = config.training.batch_size

    # Create transforms from config
    train_transform = get_transforms_from_config(
        config=config.data,
        mode="train"
    )

    val_transform = get_transforms_from_config(
        config=config.data,
        mode="val"
    )

    # Get dataset type from config (default to normal for FL)
    dataset_type = config.data.dataset_type

    # Create datasets
    train_dataset = create_adni_dataset(
        dataset_type=dataset_type,
        csv_path=config.data.train_csv_path,
        img_dir=config.data.img_dir,
        transform=train_transform,
        cache_rate=config.data.cache_rate,
        num_workers=config.data.cache_num_workers,
        cache_dir=config.data.cache_dir,
        classification_mode=config.data.classification_mode,
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
    )

    # Create data loaders with optimized multiprocessing settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.training.num_workers > 0 else None,
        multiprocessing_context=config.data.multiprocessing_context,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.training.num_workers > 0 else None,
        multiprocessing_context=config.data.multiprocessing_context,
    )

    return train_loader, val_loader


def load_config_from_yaml(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary
    """
    return Config.from_yaml(config_path)


def create_criterion(
    config: Config,
    train_dataset: Optional[torch.utils.data.Dataset] = None,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    """Create loss criterion based on configuration.

    Args:
        config: Configuration dictionary
        train_dataset: Optional training dataset for computing class weights
        device: Device to place the criterion on

    Returns:
        Loss criterion
    """
    if config.training.use_class_weights and train_dataset is not None:
        # Get labels from the training dataset
        labels = [sample["label"] for sample in train_dataset.base.data_list]

        # Compute class weights
        class_counts = {}
        for label in labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Determine number of classes
        num_classes = config.model.num_classes
        if config.data.classification_mode == "CN_AD":
            num_classes = 2

        # Calculate weights based on weight_type
        weight_type = config.training.class_weight_type

        sorted_counts = [class_counts.get(i, 0) for i in range(num_classes)]
        total_samples = sum(sorted_counts)

        if weight_type == "inverse":
            class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in sorted_counts]
        elif weight_type == "sqrt_inverse":
            class_weights = [np.sqrt(total_samples / (num_classes * count)) if count > 0 else 1.0 for count in sorted_counts]
        elif weight_type == "effective":
            beta = 0.9999
            effective_nums = [1.0 - np.power(beta, count) for count in sorted_counts]
            class_weights = [(1.0 - beta) / num if num > 0 else 1.0 for num in effective_nums]
        elif weight_type == "manual" and config.training.manual_class_weights is not None:
            class_weights = config.training.manual_class_weights
        else:
            class_weights = [1.0] * num_classes

        print(f"Class weights ({weight_type}): {class_weights}")

        weights = torch.FloatTensor(class_weights).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    else:
        return nn.CrossEntropyLoss()
