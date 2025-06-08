"""adni_flwr: A Flower-based Federated Learning framework for ADNI classification."""

import os
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix

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

    print(f"Creating model '{config.model.name}' with kwargs: {model_kwargs}")
    model = ModelFactory.create_model(config.model.name, **model_kwargs)

    # Debug the model architecture
    debug_model_architecture(model, f"{config.model.name} Model")

    return model


def safe_parameters_to_ndarrays(parameters) -> List[np.ndarray]:
    """Safely convert various parameter types to list of numpy arrays.

    Args:
        parameters: Can be Parameters object, list of numpy arrays, or other types

    Returns:
        List of numpy arrays

    Raises:
        ValueError: If the parameter type is not supported
    """
    if hasattr(parameters, 'tensors'):
        # It's a Parameters object, convert it
        from flwr.common import parameters_to_ndarrays
        param_arrays = parameters_to_ndarrays(parameters)
        print(f"Converted Parameters object to {len(param_arrays)} numpy arrays")
        return param_arrays
    elif isinstance(parameters, list):
        # Check if it's a list of numpy arrays
        if all(isinstance(p, np.ndarray) for p in parameters):
            print(f"Using provided list of {len(parameters)} numpy arrays")
            return parameters
        else:
            raise ValueError("List contains non-numpy array elements")
    else:
        raise ValueError(f"Unsupported parameter type: {type(parameters)}. "
                       f"Expected Parameters object or list of numpy arrays.")


def get_params(model: nn.Module) -> List[np.ndarray]:
    """Get model parameters as a list of NumPy arrays.

    Args:
        model: The model

    Returns:
        List of NumPy arrays representing the model parameters
    """
    model_state = model.state_dict()
    param_keys = list(model_state.keys())

    # Debug information (reduced verbosity)
    print(f"Extracting {len(param_keys)} parameters from model")

    # Extract parameters
    params = []
    for key, val in model_state.items():
        param_array = val.cpu().numpy()
        params.append(param_array)

    return params


def set_params(model: nn.Module, params: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays.

    Args:
        model: The model
        params: List of NumPy arrays representing the model parameters
    """
    # Get current model state dict keys
    model_keys = list(model.state_dict().keys())

    # Debug information
    print(f"Setting {len(params)} parameters to model (model expects {len(model_keys)})")

    # Check if the number of parameters matches
    if len(model_keys) != len(params):
        print(f"WARNING: Parameter count mismatch! Model expects {len(model_keys)} but received {len(params)}")

        # If we have fewer params than expected, something is wrong
        if len(params) < len(model_keys):
            print("ERROR: Insufficient parameters provided!")
            print(f"Model keys: {model_keys[:10]}...")  # Show first 10 keys
            raise ValueError(f"Cannot set parameters: received {len(params)} parameters but model expects {len(model_keys)}")

    # Try to create state dict with available parameters
    params_dict = zip(model_keys, params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # Try strict loading first, fall back to non-strict if it fails
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Successfully loaded parameters")
    except RuntimeError as e:
        print(f"❌ Strict loading failed: {e}")
        print("Attempting to load with strict=False...")

        # Try non-strict loading
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5

        # If there are critical missing keys, this might still fail
        if missing_keys:
            print("⚠️  WARNING: Some model parameters were not loaded!")
            print("This might cause training issues. Check model architecture consistency.")
        else:
            print("✅ Successfully loaded parameters with strict=False")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_num: int = 1,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
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
        scheduler: Optional learning rate scheduler to step after each epoch

    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.to(device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    current_lr = optimizer.param_groups[0]['lr']
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    for epoch in range(epoch_num):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

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

            # Track metrics for current epoch
            epoch_loss += loss.item() * gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            epoch_samples += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

            # Track metrics for overall training
            total_loss += loss.item() * gradient_accumulation_steps
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            batch_idx += 1

        # Calculate and log epoch metrics
        epoch_avg_loss = epoch_loss / len(train_loader)
        epoch_avg_acc = 100.0 * epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
        print(f"Epoch {epoch+1}/{epoch_num}: loss={epoch_avg_loss:.4f}, accuracy={epoch_avg_acc:.2f}%")

        # Step the scheduler after each epoch if provided
        if scheduler is not None:
            # Handle ReduceLROnPlateau scheduler which requires validation loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_avg_loss)
            else:
                scheduler.step()

            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.8f}")

    avg_loss = total_loss / (len(train_loader) * epoch_num)
    avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_accuracy, current_lr


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


def test_with_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = False,
    num_classes: int = 3
) -> Tuple[float, float, np.ndarray]:
    """Evaluate the model on the test set and generate confusion matrix.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for evaluation
        mixed_precision: Whether to use mixed precision
        num_classes: Number of classes for the confusion matrix

    Returns:
        Tuple of (average loss, average accuracy, confusion matrix)
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Store all predictions and true labels
    all_preds = []
    all_labels = []

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

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return avg_loss, avg_accuracy, cm


def test_with_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = False
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate the model on the test set and return predictions and true labels.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for evaluation
        mixed_precision: Whether to use mixed precision

    Returns:
        Tuple of (average loss, average accuracy, all predictions, all true labels)
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Store all predictions and true labels
    all_preds = []
    all_labels = []

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

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_accuracy, all_preds, all_labels


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

    # Print transform processes
    print(f"Train transform: {train_transform}")
    print(f"Val transform: {val_transform}")

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


def is_fl_client_checkpoint(checkpoint_path: str) -> bool:
    """Check if a checkpoint file is an FL client checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        True if it's an FL client checkpoint, False if it's a regular model checkpoint
    """
    try:
        if not os.path.exists(checkpoint_path):
            return False

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # FL client checkpoints have these specific keys
        fl_checkpoint_keys = ['client_id', 'round', 'training_history']

        # Check if it has FL-specific keys
        has_fl_keys = any(key in checkpoint for key in fl_checkpoint_keys)

        return has_fl_keys

    except Exception as e:
        print(f"Error checking checkpoint type: {e}")
        return False


def load_fl_client_checkpoint_to_model(checkpoint_path: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Load an FL client checkpoint and extract model state dict.

    Args:
        checkpoint_path: Path to the FL client checkpoint
        model: Model to load the state into
        device: Device for loading

    Returns:
        Dictionary containing checkpoint metadata (round, client_id, etc.)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded FL client model state from {checkpoint_path}")
        else:
            print(f"Warning: No model_state_dict found in FL checkpoint {checkpoint_path}")

        # Return metadata for potential use
        metadata = {
            'round': checkpoint.get('round', 0),
            'client_id': checkpoint.get('client_id', 'unknown'),
            'train_accuracy': checkpoint.get('train_accuracy', 0.0),
            'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
            'strategy_name': checkpoint.get('strategy_name', 'unknown'),
        }

        print(f"FL checkpoint metadata: round={metadata['round']}, "
              f"client_id={metadata['client_id']}, "
              f"strategy={metadata['strategy_name']}, "
              f"accuracy={metadata['train_accuracy']:.2f}%")

        return metadata

    except Exception as e:
        print(f"Error loading FL client checkpoint: {e}")
        return {}


def handle_pretrained_checkpoint(config: Config, model: torch.nn.Module, device: torch.device) -> dict:
    """Handle loading of pretrained checkpoint (either regular model or FL client checkpoint).

    Args:
        config: Configuration object
        model: Model to load checkpoint into
        device: Device for loading

    Returns:
        Dictionary containing checkpoint metadata (empty for regular checkpoints)
    """
    if not config.model.pretrained_checkpoint:
        return {}

    checkpoint_path = config.model.pretrained_checkpoint

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return {}

    # Detect checkpoint type
    if is_fl_client_checkpoint(checkpoint_path):
        print(f"Detected FL client checkpoint: {checkpoint_path}")
        return load_fl_client_checkpoint_to_model(checkpoint_path, model, device)
    else:
        print(f"Detected regular model checkpoint: {checkpoint_path}")
        # Load as regular model state dict
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)

            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                # Training checkpoint format
                model.load_state_dict(state_dict['model_state_dict'])
                print(f"Loaded model state dict from training checkpoint: {checkpoint_path}")
            else:
                # Direct state dict format
                model.load_state_dict(state_dict)
                print(f"Loaded model state dict: {checkpoint_path}")

        except Exception as e:
            print(f"Error loading regular checkpoint: {e}")

        return {}  # No metadata for regular checkpoints


def debug_model_architecture(model: nn.Module, model_name: str = "Model") -> None:
    """Debug model architecture by printing layer structure.

    Args:
        model: The model to debug
        model_name: Name for debugging output
    """
    print(f"\n=== {model_name} Architecture ===")
    print(f"Model type: {type(model).__name__}")

    # Print state dict summary
    state_dict = model.state_dict()
    print(f"Total parameters: {len(state_dict)}")
    print(f"Parameter keys: {list(state_dict.keys())[:5]}..." if len(state_dict) > 5 else f"Parameter keys: {list(state_dict.keys())}")
    print("=" * 40)


def verify_model_consistency(model1: nn.Module, model2: nn.Module,
                           name1: str = "Model1", name2: str = "Model2") -> bool:
    """Verify that two models have consistent architectures.

    Args:
        model1: First model
        model2: Second model
        name1: Name of first model
        name2: Name of second model

    Returns:
        True if models are consistent, False otherwise
    """
    print(f"\n=== Model Consistency Check: {name1} vs {name2} ===")

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # Check if keys match
    if keys1 != keys2:
        print("❌ Models have different parameter keys!")

        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1

        if only_in_1:
            print(f"Keys only in {name1}: {sorted(only_in_1)}")
        if only_in_2:
            print(f"Keys only in {name2}: {sorted(only_in_2)}")

        return False

    # Check if shapes match
    shape_mismatch = False
    for key in keys1:
        shape1 = state_dict1[key].shape
        shape2 = state_dict2[key].shape
        if shape1 != shape2:
            print(f"❌ Shape mismatch for {key}: {shape1} vs {shape2}")
            shape_mismatch = True

    if shape_mismatch:
        return False

    print("✅ Models have consistent architectures!")
    return True
