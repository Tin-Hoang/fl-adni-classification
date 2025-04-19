"""Transforms module for ADNI MRI classification datasets."""

from typing import Tuple, Optional, Union
import torch
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandAffined,
    ToTensord,
    Resized,
    Rand3DElasticd,
    RandGaussianNoised,
    RandAdjustContrastd,
)


def get_transforms(mode: str = "train",
                  resize_size: Tuple[int, int, int] = (160, 160, 160),
                  resize_mode: str = "trilinear",
                  use_spacing: bool = True,
                  spacing_size: Tuple[float, float, float] = (1.5, 1.5, 1.5),
                  device: Optional[Union[str, torch.device]] = None) -> monai.transforms.Compose:
    """Get transforms for training or validation.

    Args:
        mode: Either "train" or "val"
        resize_size: Tuple of (height, width, depth) for resizing
        resize_mode: Interpolation mode for resizing
        use_spacing: Whether to include the Spacing transform (default: True)
        spacing_size: Tuple of (x, y, z) spacing in mm (default: (1.5, 1.5, 1.5))
        device: Device to use for transforms (default: None, will use CPU)

    Returns:
        A Compose transform
    """
    # Ensure resize_size is a tuple for consistency
    if not isinstance(resize_size, tuple):
        resize_size = tuple(resize_size)

    # Ensure spacing_size is a tuple
    if not isinstance(spacing_size, tuple):
        spacing_size = tuple(spacing_size)

    common_transforms = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),  # Use specific orientation to ensure consistency
    ]

    # Add spacing transform if requested
    if use_spacing:
        common_transforms.append(
            # Ensure consistent spacing
            Spacingd(keys=["image"], pixdim=spacing_size, mode="bilinear")
        )

    # Add the rest of the transforms
    common_transforms.extend([
        # More robust intensity scaling using percentiles
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0,
            a_max=100.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Ensure all images have the same size
        Resized(
            keys=["image"],
            spatial_size=resize_size,
            mode=resize_mode,
        ),
    ])

    if mode == "train":
        train_transforms = [
            # Existing augmentations
            RandAffined(
                keys=["image"],
                prob=0.8,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.2, 0.2, 0.2),
                mode="bilinear",
                padding_mode="zeros",
                device=device,
            ),

            # Add elastic deformations
            Rand3DElasticd(
                keys=["image"],
                prob=0.3,
                sigma_range=(5, 8),
                magnitude_range=(0.1, 0.3),
                spatial_size=resize_size,  # Use the same resize_size from parameters
                mode="bilinear",
                padding_mode="zeros",
                device=device,
            ),

            # Add intensity augmentations
            RandGaussianNoised(
                keys=["image"],
                prob=0.5,
                mean=0.0,
                std=0.1,
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.8, 1.2),
            ),

            # Convert to tensor
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(common_transforms + train_transforms)
    else:  # val
        val_transforms = [
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(common_transforms + val_transforms)
