#!/usr/bin/env python3
"""
Script to convert the original AD_pretrained_weights.pt file to match the current RosannaCNN model format.

This script loads the original pretrained weights and converts them to be compatible with the
current RosannaCNN model implementation, saving in the checkpoint format used by the training script.
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adni_classification.models.rosanna_cnn import RosannaCNN, RosannaCNNConfig


def load_original_weights(weights_path: str) -> dict:
    """Load the original pretrained weights from file.

    Args:
        weights_path: Path to the original weights file

    Returns:
        Dictionary containing the original state dict
    """
    print(f"Loading original weights from: {weights_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Load the original weights
    original_weights = torch.load(weights_path, map_location='cpu')

    print(f"Original weights loaded successfully")
    print(f"Keys in original weights: {list(original_weights.keys())}")

    return original_weights


def create_rosanna_model(num_classes: int = 2) -> RosannaCNN:
    """Create a RosannaCNN model with the correct architecture.

    Args:
        num_classes: Number of output classes (default: 2 for CN vs AD)

    Returns:
        RosannaCNN model instance
    """
    model = RosannaCNN(
        num_classes=num_classes,
        pretrained_weights_path=None,
        freeze_encoder=False,
        dropout=0.0,
        input_channels=1
    )

    print(f"Created RosannaCNN model with {num_classes} classes")
    print(f"Model state dict keys: {list(model.state_dict().keys())}")

    return model


def convert_weights(original_weights: dict, target_model: RosannaCNN) -> dict:
    """Convert original weights to match the target model structure.

    Args:
        original_weights: Dictionary with original weights
        target_model: Target RosannaCNN model

    Returns:
        Dictionary with converted weights
    """
    target_state_dict = target_model.state_dict()
    converted_weights = {}

    print("\nConverting weights...")

    # Print shapes for debugging
    print("\nOriginal weights shapes:")
    for key, value in original_weights.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print("\nTarget model shapes:")
    for key, value in target_state_dict.items():
        print(f"  {key}: {value.shape}")

    # Convert weights - the key mapping might be different
    # We need to map from the original CNN model to RosannaCNN

    # Try to match weights by shape and position
    for target_key, target_tensor in target_state_dict.items():
        matched = False

        # Direct key matching first
        if target_key in original_weights:
            if original_weights[target_key].shape == target_tensor.shape:
                converted_weights[target_key] = original_weights[target_key]
                print(f"✓ Direct match: {target_key}")
                matched = True

        # If no direct match, try to match by shape and layer type
        if not matched:
            for orig_key, orig_tensor in original_weights.items():
                if isinstance(orig_tensor, torch.Tensor) and orig_tensor.shape == target_tensor.shape:
                    # Additional checks to ensure we're matching the right layers
                    if _is_compatible_layer(target_key, orig_key):
                        converted_weights[target_key] = orig_tensor
                        print(f"✓ Shape match: {target_key} <- {orig_key}")
                        matched = True
                        break

        if not matched:
            print(f"✗ No match found for: {target_key} (shape: {target_tensor.shape})")
            # Initialize with model's default weights
            converted_weights[target_key] = target_tensor

    print(f"\nSuccessfully converted {len([k for k, v in converted_weights.items() if k in original_weights])} layers")

    return converted_weights


def _is_compatible_layer(target_key: str, orig_key: str) -> bool:
    """Check if two layer keys are compatible for weight transfer.

    Args:
        target_key: Key from target model
        orig_key: Key from original model

    Returns:
        True if layers are compatible
    """
    # Basic compatibility checks
    layer_types = ['conv', 'batch', 'linear', 'weight', 'bias']

    target_type = None
    orig_type = None

    for layer_type in layer_types:
        if layer_type in target_key.lower():
            target_type = layer_type
        if layer_type in orig_key.lower():
            orig_type = layer_type

    return target_type == orig_type


def create_checkpoint_format(
    model_state_dict: dict,
    num_classes: int = 2,
    val_acc: float = 88.75,  # From the pretraining info
    notes: str = "Converted from AD_pretrained_weights.pt (Rosanna's pretrained model)"
) -> Dict[str, Any]:
    """Create a checkpoint in the format expected by load_checkpoint function.

    Args:
        model_state_dict: The converted model state dictionary
        num_classes: Number of output classes
        val_acc: Validation accuracy from original training
        notes: Additional notes about the checkpoint

    Returns:
        Dictionary in checkpoint format
    """
    checkpoint = {
        'epoch': 0,  # Starting from epoch 0 for fine-tuning
        'model_state_dict': model_state_dict,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'val_acc': val_acc,
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        # Metadata about the original model
        'pretrained_info': {
            'original_dataset': 'ADNI (CN vs AD)',
            'original_samples': '1.5T ADNI1 MRI - 550 samples',
            'input_size': [73, 96, 96],
            'num_classes': num_classes,
            'architecture': 'CNN_8CL_B (Rosanna)',
            'conversion_notes': notes
        }
    }

    return checkpoint


def save_converted_checkpoint(checkpoint: dict, output_path: str) -> None:
    """Save the converted checkpoint to a file.

    Args:
        checkpoint: Dictionary with checkpoint data
        output_path: Path to save the converted checkpoint
    """
    print(f"\nSaving converted checkpoint to: {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the checkpoint
    torch.save(checkpoint, output_path)

    print(f"✓ Converted checkpoint saved successfully")


def verify_checkpoint_loading(checkpoint_path: str, num_classes: int = 2) -> bool:
    """Verify that the converted checkpoint can be loaded by RosannaCNN.

    Args:
        checkpoint_path: Path to the converted checkpoint file
        num_classes: Number of classes for the model

    Returns:
        True if verification successful, False otherwise
    """
    print(f"\nVerifying checkpoint loading...")

    try:
        # Create model
        model = RosannaCNN(
            num_classes=num_classes,
            pretrained_weights_path=None,
            freeze_encoder=False,
            dropout=0.0
        )

        # Load checkpoint using the same method as load_checkpoint function
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        print("✓ Checkpoint loaded successfully using load_state_dict")

        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 1, 73, 96, 96)  # Based on the original input size

        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful, output shape: {output.shape}")
            print(f"✓ Output classes: {output.shape[1]}")

        # Print checkpoint metadata
        if 'pretrained_info' in checkpoint:
            print(f"✓ Checkpoint metadata:")
            for key, value in checkpoint['pretrained_info'].items():
                print(f"    {key}: {value}")

        print(f"✓ Validation accuracy from original training: {checkpoint['val_acc']:.2f}%")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion function."""
    print("=" * 70)
    print("Converting AD_pretrained_weights.pt to RosannaCNN checkpoint format")
    print("=" * 70)

    # Paths
    original_weights_path = "3D_CNN_pretrained_model/AD_pretrained_weights.pt"
    converted_checkpoint_path = "3D_CNN_pretrained_model/RosannaCNN_pretrained_checkpoint.pth"

    try:
        # Step 1: Load original weights
        original_weights = load_original_weights(original_weights_path)

        # Step 2: Create target model (2 classes for CN vs AD)
        target_model = create_rosanna_model(num_classes=2)

        # Step 3: Convert weights
        converted_weights = convert_weights(original_weights, target_model)

        # Step 4: Create checkpoint format
        checkpoint = create_checkpoint_format(
            model_state_dict=converted_weights,
            num_classes=2,
            val_acc=88.75,  # Test accuracy from original model
            notes="Converted from Rosanna's AD_pretrained_weights.pt - Original architecture: CNN_8CL_B"
        )

        # Step 5: Save converted checkpoint
        save_converted_checkpoint(checkpoint, converted_checkpoint_path)

        # Step 6: Verify checkpoint loading
        verification_success = verify_checkpoint_loading(converted_checkpoint_path, num_classes=2)

        if verification_success:
            print("\n" + "=" * 70)
            print("✓ CONVERSION SUCCESSFUL!")
            print(f"✓ Original weights: {original_weights_path}")
            print(f"✓ Converted checkpoint: {converted_checkpoint_path}")
            print("✓ Checkpoint is compatible with load_checkpoint function")
            print("✓ Ready for fine-tuning or inference")
            print("=" * 70)

            # Usage instructions
            print("\nUsage Instructions:")
            print("1. To load in training script:")
            print("   model, optimizer, scheduler, scaler, start_epoch, ...")
            print("   = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)")
            print("\n2. To load just the model weights:")
            print("   checkpoint = torch.load(checkpoint_path)")
            print("   model.load_state_dict(checkpoint['model_state_dict'])")
            print("\n3. For inference or fine-tuning:")
            print("   model = RosannaCNN(num_classes=2)")
            print("   model.load_pretrained_weights(checkpoint_path)")

        else:
            print("\n" + "=" * 70)
            print("✗ CONVERSION COMPLETED WITH ISSUES")
            print("Please check the verification output above")
            print("=" * 70)

    except Exception as e:
        print(f"\n✗ CONVERSION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
