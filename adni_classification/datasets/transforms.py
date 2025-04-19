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


def test_transforms():
    """Test the transformation pipeline on sample images from the ADNI dataset.

    This function loads a few sample images from the dataset and applies the transforms,
    displaying information about the final transformed images.

    Example usage:
        python -m adni_classification.datasets.transforms \
            --csv_path data/ADNI/train.csv \
            --img_dir data/ADNI/images \
            --num_samples 3 \
            --visualize
    """
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path

    # Import here to avoid circular imports
    from adni_classification.datasets.adni_cache_dataset import ADNICacheDataset

    parser = argparse.ArgumentParser(description="Test the ADNI dataset transforms")
    parser.add_argument("--csv_path", type=str,
                        default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_594images_client_all_train_475images.csv",
                        help="Path to the CSV file containing image metadata and labels")
    parser.add_argument("--img_dir", type=str,
                        default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_611images_step3_skull_stripping",
                        help="Path to the directory containing the image files")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to test")
    parser.add_argument("--resize_size", type=str, default="182,218,182",
                        help="Resize dimensions (height,width,depth)")
    parser.add_argument("--use_spacing", type=str, choices=["true", "false"], default="true",
                        help="Whether to include spacing transform (true/false)")
    parser.add_argument("--spacing_size", type=str, default="1.5,1.5,1.5",
                        help="Spacing dimensions (x,y,z) in mm")
    parser.add_argument("--cache_rate", type=float, default=1.0,
                        help="Percentage of data to cache (0.0-1.0)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the transformed images")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for transforms (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--dataset_type", type=str, default="cache", choices=["cache", "normal"],
                        help="Type of dataset to use for testing")
    args = parser.parse_args()

    # Parse resize dimensions
    resize_size = tuple(map(int, args.resize_size.split(',')))

    # Parse spacing parameters
    use_spacing = args.use_spacing.lower() == "true"
    spacing_size = tuple(map(float, args.spacing_size.split(',')))

    # Parse device
    device = torch.device(args.device) if args.device else None

    print(f"Testing transforms with resize size: {resize_size}")
    print(f"Using spacing transform: {use_spacing}")
    if use_spacing:
        print(f"Spacing size: {spacing_size}")
    print(f"CSV path: {args.csv_path}")
    print(f"Image directory: {args.img_dir}")
    print(f"Cache rate: {args.cache_rate}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Device: {device}")
    print(f"Dataset type: {args.dataset_type}")

    # Create transforms for testing
    test_transforms = get_transforms(
        mode="val",
        resize_size=resize_size,
        resize_mode="trilinear",
        use_spacing=use_spacing,
        spacing_size=spacing_size,
        device=device
    )

    # Dynamically import the right dataset class
    if args.dataset_type == "cache":
        from adni_classification.datasets.adni_cache_dataset import ADNICacheDataset as DatasetClass
    else:
        from adni_classification.datasets.adni_dataset import ADNIDataset as DatasetClass

    # Create a dataset without transforms first
    try:
        if args.dataset_type == "cache":
            dataset = DatasetClass(
                args.csv_path,
                args.img_dir,
                cache_rate=args.cache_rate,
                num_workers=args.num_workers
            )
        else:
            dataset = DatasetClass(
                args.csv_path,
                args.img_dir
            )

        print(f"Successfully created dataset with {len(dataset)} samples")

        # Test transforms on a few samples
        print(f"\nTesting transforms on {args.num_samples} samples...")

        for i in range(min(args.num_samples, len(dataset))):
            # Get the sample without transforms
            sample = dataset[i]
            image_path = sample["image"]
            label = sample["label"]
            label_name = [k for k, v in dataset.label_map.items() if v == label][0]

            print(f"\nSample {i+1}: {Path(image_path).name}, Label: {label_name} ({label})")

            # Apply transforms
            print("Applying transforms...")
            transformed = test_transforms({"image": image_path, "label": label})

            # Get transformed image shape
            transformed_image = transformed["image"]
            if isinstance(transformed_image, np.ndarray):
                print(f"Transformed image: NumPy array with shape {transformed_image.shape}")
            else:
                print(f"Transformed image: Tensor with shape {transformed_image.shape} on {transformed_image.device}")

            # Visualize if requested
            if args.visualize:
                try:
                    # If it's a tensor, convert to numpy
                    if hasattr(transformed_image, 'detach'):
                        img_data = transformed_image.detach().cpu().numpy()
                    else:
                        img_data = transformed_image

                    # Take middle slices
                    if len(img_data.shape) == 4:  # [C, H, W, D]
                        img_data = img_data[0]  # Take first channel

                    mid_z = img_data.shape[2] // 2
                    mid_y = img_data.shape[1] // 2
                    mid_x = img_data.shape[0] // 2

                    # Create a figure with three subplots (one for each view)
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # Axial view (top-down)
                    axes[0].imshow(img_data[:, :, mid_z], cmap='gray')
                    axes[0].set_title(f'Axial (z={mid_z})')

                    # Coronal view (front-back)
                    axes[1].imshow(img_data[:, mid_y, :], cmap='gray')
                    axes[1].set_title(f'Coronal (y={mid_y})')

                    # Sagittal view (side)
                    axes[2].imshow(img_data[mid_x, :, :], cmap='gray')
                    axes[2].set_title(f'Sagittal (x={mid_x})')

                    # Add overall title
                    plt.suptitle(f'Sample {i+1}: {Path(image_path).name} - {label_name} ({label})')

                    # Save the figure
                    output_dir = Path('transform_test_output')
                    output_dir.mkdir(exist_ok=True)
                    plt.savefig(output_dir / f'sample_{i+1}_{Path(image_path).stem}.png')
                    plt.close()
                    print(f"Visualization saved to transform_test_output/sample_{i+1}_{Path(image_path).stem}.png")
                except Exception as e:
                    print(f"Error visualizing image: {e}")

        print("\nTransform test completed successfully.")

    except ValueError as e:
        print(f"\nError creating dataset: {e}")


# Allow direct execution of this module
if __name__ == "__main__":
    test_transforms()
