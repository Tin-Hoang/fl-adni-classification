"""Dataset module for ADNI classification."""

import os
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Resized,
    Lambdad,
)


# Define these functions at the module level so they can be pickled
def debug_print_stage(x, stage="DEBUG"):
    """Print debug information about an image tensor at a specific stage.

    Args:
        x: Image tensor
        stage: Processing stage description

    Returns:
        x: The original image tensor (unchanged)
    """
    print(f"{stage} Image shape: {x.shape}")
    return x


def debug_original(x):
    """Debug function for original image."""
    return debug_print_stage(x, "ORIGINAL")


def debug_orientation(x):
    """Debug function after orientation transformation."""
    return debug_print_stage(x, "AFTER ORIENTATION")


def debug_spacing(x):
    """Debug function after spacing transformation."""
    return debug_print_stage(x, "AFTER SPACING")


def debug_resize(x):
    """Debug function after resize transformation."""
    return debug_print_stage(x, "AFTER RESIZE")


# Function for shape debug that can be pickled (module-level)
def shape_debug_func(x, expected_size):
    """Debug function to check tensor shape.

    Args:
        x: Image tensor
        expected_size: Expected size of the tensor

    Returns:
        x: The original or resized image tensor
    """
    current_shape = x.shape[1:]  # Get spatial dimensions

    if current_shape != expected_size:
        print(f"Warning: Shape mismatch before correction. Current: {current_shape}, Expected: {expected_size}")
        # Ensure shape matches by force resize
        resize = monai.transforms.Resize(spatial_size=expected_size)
        x = resize(x)
        print(f"After resize: {x.shape[1:]}")
    return x


# Create a wrapper class for shape_debug_func with fixed expected_size
class ShapeCheckerFunction:
    """Class that checks and corrects image shape, designed to be picklable.

    Args:
        expected_size: Expected size tuple
    """
    def __init__(self, expected_size):
        self.expected_size = expected_size

    def __call__(self, x):
        """Check and correct the image shape.

        Args:
            x: Input image tensor

        Returns:
            Image tensor with corrected shape
        """
        return shape_debug_func(x, self.expected_size)

    def __reduce__(self):
        """Support pickling by returning a tuple of class, args for reconstruction."""
        return (self.__class__, (self.expected_size,))


# Create a custom shape checker that can be pickled
class ShapeChecker:
    """Shape checking transform that can be pickled."""

    def __init__(self, expected_size):
        self.expected_size = expected_size

    def __call__(self, image):
        """Check that the image has the expected shape.

        Args:
            image: The image tensor to check

        Returns:
            The original image or resized image if needed
        """
        # Get the spatial dimensions (excluding channel dimension)
        spatial_shape = image.shape[1:]
        expected_shape = tuple(self.expected_size)

        # Only resize if shapes don't match (without debug prints)
        if spatial_shape != expected_shape:
            if len(spatial_shape) == len(expected_shape):
                # Use monai's Resize transform to fix the shape
                resize = monai.transforms.Resize(spatial_size=expected_shape)
                return resize(image)

        return image


class ADNIDataset(Dataset):
    """Dataset for ADNI MRI classification.

    This dataset loads 3D MRI images from the ADNI dataset and their corresponding labels.
    The dataset supports two CSV formats:

    Original format:
    - Image Data ID: The ID of the image in the ADNI database, prefixed with 'I'
    - Subject: The subject ID (e.g., "136_S_1227")
    - Group: The diagnosis group (AD, MCI, CN)
    - Description: The image description (e.g., "MPR; ; N3; Scaled")

    Alternative format:
    - image_id: The ID of the image in the ADNI database (without 'I' prefix)
    - DX: The diagnosis group (Dementia, MCI, CN)

    The image files are expected to be in NiFTI format (.nii or .nii.gz) and organized in a directory structure
    where the Image ID appears both in the filename (as a suffix before .nii/.nii.gz) and in the parent directory.
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[monai.transforms.Compose] = None,
    ):
        """Initialize the dataset.

        Args:
            csv_path: Path to the CSV file containing image metadata and labels
            img_dir: Path to the directory containing the image files
            transform: Optional transform to apply to the images
        """
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transform = transform

        # Load the CSV file
        self.data = pd.read_csv(csv_path)

        # Detect CSV format
        self.csv_format = self._detect_csv_format()

        # Map diagnosis groups to numeric labels based on detected format
        self.label_map = {"CN": 0, "MCI": 1, "AD": 2, "Dementia": 2}

        # Filter out rows with missing labels and standardize data
        self._standardize_data()

        # Create a mapping from Image Data ID to file path
        self.image_paths = self._find_image_files(img_dir)

        # Filter out rows with missing image files
        csv_image_ids = set(self.data["Image Data ID"].unique())
        mapped_image_ids = set(self.image_paths.keys())

        # Find missing IDs (IDs in CSV not found in image files)
        missing_ids = csv_image_ids - mapped_image_ids

        if missing_ids:
            missing_count = len(missing_ids)
            examples = list(missing_ids)[:5]
            example_str = ", ".join(examples)
            if len(missing_ids) > 5:
                example_str += f", and {missing_count - 5} more"

            error_msg = (
                f"Error: {missing_count} Image IDs from the CSV could not be found in the image files.\n"
                f"Examples: {example_str}\n"
                f"Please ensure that all Image IDs in the CSV have corresponding image files."
            )
            raise ValueError(error_msg)

        # Keep only rows with valid image files
        self.data = self.data[self.data["Image Data ID"].isin(self.image_paths.keys())]

        print(f"Found {len(self.image_paths)} image files in {img_dir}")
        print(f"Final dataset size: {len(self.data)} samples")

    def _detect_csv_format(self) -> str:
        """Detect the format of the CSV file.

        Returns:
            String indicating the detected format: "original" or "alternative"
        """
        if "DX" in self.data.columns and "image_id" in self.data.columns:
            print("\n" + "="*80)
            print("Detected ALTERNATIVE CSV format with:")
            print("- 'DX' column for diagnosis (Dementia, MCI, CN)")
            print("- 'image_id' column for image identifiers (without 'I' prefix)")
            print("="*80 + "\n")
            return "alternative"
        elif "Group" in self.data.columns and "Image Data ID" in self.data.columns:
            print("\n" + "="*80)
            print("Detected ORIGINAL CSV format with:")
            print("- 'Group' column for diagnosis (AD, MCI, CN)")
            print("- 'Image Data ID' column for image identifiers (with 'I' prefix)")
            print("="*80 + "\n")
            return "original"
        else:
            raise ValueError("Unknown CSV format. CSV must have either 'Group' and 'Image Data ID' columns (original format) or 'DX' and 'image_id' columns (alternative format).")

    def _standardize_data(self) -> None:
        """Standardize the data based on the detected CSV format."""
        if self.csv_format == "original":
            # Filter out rows with missing labels
            self.data = self.data[self.data["Group"].isin(["AD", "MCI", "CN"])]
        else:  # alternative format
            # Map DX column values to standard Group values
            dx_to_group = {"Dementia": "AD", "MCI": "MCI", "CN": "CN"}

            # Filter out rows with missing or invalid labels
            self.data = self.data[self.data["DX"].isin(list(dx_to_group.keys()))]

            # Create standardized columns
            self.data["Group"] = self.data["DX"].map(dx_to_group)

            # Convert image_id to string to ensure proper handling
            self.data["image_id"] = self.data["image_id"].astype(str)

            # Add 'I' prefix to image_id to create Image Data ID if not already prefixed
            # Also strip any decimal points (e.g., "42832.0" -> "42832")
            self.data["Image Data ID"] = self.data["image_id"].apply(
                lambda x: f"I{x.split('.')[0]}" if not x.startswith('I') else x.split('.')[0]
            )

        # Verify that we have data after filtering
        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {self.csv_path} after filtering. "
                             f"Check that the CSV contains the expected columns and values.")

        # Print summary of the standardized data
        print(f"CSV format: {self.csv_format}")
        print(f"Total samples after standardization: {len(self.data)}")
        print("Group distribution:")
        for group, count in self.data["Group"].value_counts().items():
            print(f"  {group}: {count}")

    def _find_image_files(self, root_dir: str) -> Dict[str, str]:
        """Find all .nii and .nii.gz files in the root directory and map them to Image IDs.

        Args:
            root_dir: Root directory to search for .nii and .nii.gz files

        Returns:
            Dictionary mapping Image IDs to file paths
        """
        image_paths = {}
        found_ids = []  # For debugging

        # Walk through the directory tree
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    file_path = os.path.join(root, file)
                    parent_dir = os.path.basename(root)

                    # For alternative format, prioritize the parent directory ID
                    if self.csv_format == "alternative":
                        # Try to get image ID from parent directory first for alternative format
                        if parent_dir.startswith('I') and parent_dir[1:].isdigit():
                            image_id = parent_dir
                        elif parent_dir.isdigit():
                            image_id = f"I{parent_dir}"
                        else:
                            # Fallback to filename ID extraction
                            image_id = self._extract_id_from_filename(file)
                    else:
                        # Original format - extract from filename
                        image_id = self._extract_id_from_filename(file)

                        # If not found in filename, try parent directory
                        if image_id is None:
                            if parent_dir.startswith('I') and parent_dir[1:].isdigit():
                                image_id = parent_dir

                    # Skip if no image ID found
                    if image_id is None:
                        continue

                    # Add to debugging list
                    if len(found_ids) < 10:
                        found_ids.append((image_id, file_path, parent_dir))

                    # Add to the mapping - if both .nii and .nii.gz exist for same ID,
                    # prioritize .nii.gz as it's likely the more recent/compressed version
                    if image_id not in image_paths or file.endswith('.nii.gz'):
                        image_paths[image_id] = file_path

        # Print the first 10 IDs found for debugging
        print("\nFirst 10 image IDs found in files:")
        for i, (id_val, path, dir_name) in enumerate(found_ids):
            print(f"{i+1}. ID: {id_val}, Directory: {dir_name}, Path: {os.path.basename(path)}")

        return image_paths

    def _extract_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract Image ID from filename.

        Args:
            filename: The filename to extract the Image ID from

        Returns:
            The extracted Image ID or None if not found
        """
        # Remove .nii.gz or .nii extension before splitting
        if filename.endswith('.nii.gz'):
            filename = filename[:-7]  # Remove .nii.gz
        elif filename.endswith('.nii'):
            filename = filename[:-4]  # Remove .nii

        parts = filename.split('_')

        # First try to find an ID that starts with 'I' followed by digits
        for part in reversed(parts):
            if part.startswith('I') and part[1:].isdigit():
                return part

        # If not found, look for a part that's just digits (for alternative format)
        if self.csv_format == "alternative":
            for part in reversed(parts):
                if part.isdigit():
                    # Add 'I' prefix to match the standardized IDs in the dataset
                    return f"I{part}"

        return None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            A dictionary containing the image and label
        """
        row = self.data.iloc[idx]
        image_id = row["Image Data ID"]
        label = self.label_map[row["Group"]]

        # Get the image path
        image_path = self.image_paths[image_id]

        # Create a dictionary with the image path and label
        data_dict = {
            "image": image_path,
            "label": label,
        }

        # Apply transforms if any
        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict


def get_transforms(mode: str = "train",
                resize_size: Tuple[int, int, int] = (160, 160, 160),
                resize_mode: str = "trilinear",
                use_spacing: bool = True,
                spacing_size: Tuple[float, float, float] = (1.5, 1.5, 1.5)) -> monai.transforms.Compose:
    """Get transforms for training or validation.

    Args:
        mode: Either "train" or "val"
        resize_size: Tuple of (height, width, depth) for resizing
        resize_mode: Interpolation mode for resizing
        use_spacing: Whether to include the Spacing transform (default: True)
        spacing_size: Tuple of (x, y, z) spacing in mm (default: (1.5, 1.5, 1.5))

    Returns:
        A Compose transform
    """
    # Ensure resize_size is a tuple for consistency
    if not isinstance(resize_size, tuple):
        resize_size = tuple(resize_size)

    # Ensure spacing_size is a tuple
    if not isinstance(spacing_size, tuple):
        spacing_size = tuple(spacing_size)

    # Create a shape checker instance that can be pickled
    shape_checker = ShapeChecker(resize_size)

    # Create a function for shape checking that can be pickled
    shape_check_func = ShapeCheckerFunction(resize_size)

    common_transforms = [
        LoadImaged(keys=["image"], image_only=False),
        # EnsureChannelFirstd(keys=["image"]),
        # Use specific orientation to ensure consistency
        # Orientationd(keys=["image"], axcodes="RAS"),
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
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=0.0,
        #     a_max=100.0,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        # Ensure all images have the same size
        Resized(
            keys=["image"],
            spatial_size=resize_size,
            mode=resize_mode,
        ),
        # Add an explicit transform to ensure consistent dimensions
        # This will guarantee that all tensors have the exact same shape
        # Lambdad(
        #     keys=["image"],
        #     func=shape_check_func,
        # ),
    ])

    if mode == "train":
        train_transforms = [
            # Stronger augmentation for small dataset
            # RandAffined(
            #     keys=["image"],
            #     prob=0.8,
            #     rotate_range=(0.1, 0.1, 0.1),
            #     scale_range=(0.2, 0.2, 0.2),
            #     mode="bilinear",
            #     padding_mode="zeros",  # Use 'zeros' for consistent behavior
            # ),
            # RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            # RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            # RandRotate90d(keys=["image"], prob=0.5, spatial_axes=[0, 1]),
            # monai.transforms.RandGaussianNoised(
            #     keys=["image"],
            #     prob=0.5,
            #     mean=0.0,
            #     std=0.1,
            # ),
            # Ensure consistent dimensions after augmentation
            # Resized(
            #     keys=["image"],
            #     spatial_size=resize_size,
            #     mode=resize_mode,
            # ),
            # Additional shape check after augmentation
            # Lambdad(
            #     keys=["image"],
            #     func=shape_check_func,
            # ),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(common_transforms + train_transforms)
    else:  # val
        val_transforms = [
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(common_transforms + val_transforms)


def test_image_path_mapping():
    """Test the image path mapping logic.

    This function creates a test dataset and prints information about the mapped image paths.
    It can be run directly to verify that the image path mapping logic is working correctly.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test the ADNI dataset image path mapping")
    parser.add_argument("--csv_path", type=str, default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_594images_client_all_train_475images.csv",
                        help="Path to the CSV file containing image metadata and labels")
    parser.add_argument("--img_dir", type=str, default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_611images_step3_skull_stripping",
                        help="Path to the directory containing the image files")
    parser.add_argument("--csv_format", type=str, choices=["original", "alternative"],
                        help="CSV format to test explicitly (will be auto-detected if not specified)")
    args = parser.parse_args()

    print("Testing ADNI dataset image path mapping...")
    print(f"CSV path: {args.csv_path}")
    print(f"Image directory: {args.img_dir}")

    # Read and print the first few rows of the CSV
    df = pd.read_csv(args.csv_path)
    print("\nFirst 10 rows of the CSV file:")
    print(df.head(10))

    # Print ID columns specifically for clarity
    if 'Image Data ID' in df.columns:
        print("\nFirst 10 Image Data IDs in CSV:")
        for i, img_id in enumerate(df['Image Data ID'].head(10)):
            print(f"{i+1}. {img_id}")

    if 'image_id' in df.columns:
        print("\nFirst 10 image_ids in CSV:")
        for i, img_id in enumerate(df['image_id'].head(10)):
            print(f"{i+1}. {img_id}")

    try:
        # Create a dataset with ID validation
        print("\nAttempting to create dataset with strict ID validation...")
        dataset = ADNIDataset(args.csv_path, args.img_dir)

        # If successful, print information about it
        print(f"Successfully created dataset with {len(dataset)} samples")
        print(f"Detected CSV format: {dataset.csv_format}")

        # Print information about the mapped image paths
        print("\nImage path mapping:")
        file_formats = {'.nii': 0, '.nii.gz': 0}

        for image_id, file_path in list(dataset.image_paths.items())[:5]:  # Show first 5 mappings
            # Determine file format
            if file_path.endswith('.nii.gz'):
                file_formats['.nii.gz'] += 1
                format_str = "(.nii.gz)"
            elif file_path.endswith('.nii'):
                file_formats['.nii'] += 1
                format_str = "(.nii)"
            else:
                format_str = "(unknown format)"

            # Get the label for this image ID
            row = dataset.data[dataset.data["Image Data ID"] == image_id]
            if not row.empty:
                label_group = row["Group"].iloc[0]
                label_idx = dataset.label_map[label_group]
                print(f"  {image_id} -> {os.path.basename(file_path)} {format_str} (Label: {label_group}, ID: {label_idx})")
            else:
                print(f"  {image_id} -> {os.path.basename(file_path)} {format_str} (Label: unknown)")

        if len(dataset.image_paths) > 5:
            print(f"  ... and {len(dataset.image_paths) - 5} more")

        # Count all file formats
        for img_path in dataset.image_paths.values():
            if img_path.endswith('.nii.gz'):
                file_formats['.nii.gz'] += 1
            elif img_path.endswith('.nii'):
                file_formats['.nii'] += 1

        print("\nFile format distribution:")
        print(f"  .nii files: {file_formats['.nii']}")
        print(f"  .nii.gz files: {file_formats['.nii.gz']}")

        # If using the alternative format, show the mapping from DX to Group
        if dataset.csv_format == "alternative":
            print("\nDX to Group mapping:")
            dx_counts = dataset.data.groupby(["DX", "Group"]).size().reset_index(name="count")
            for _, row in dx_counts.iterrows():
                print(f"  {row['DX']} -> {row['Group']}: {row['count']} samples")

        print("\nFinal dataset summary:")
        print(f"Total images found: {len(dataset.image_paths)}")
        print(f"Total samples in dataset: {len(dataset)}")

    except ValueError as e:
        print(f"\nError creating dataset: {e}")

    print("\nTest completed.")


def test_transforms():
    """Test the transformation pipeline on sample images from the ADNI dataset.

    This function loads a few sample images from the dataset and applies the transforms,
    displaying information about the final transformed images.
    """
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

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
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the transformed images")
    args = parser.parse_args()

    # Parse resize dimensions
    resize_size = tuple(map(int, args.resize_size.split(',')))

    # Parse spacing parameters
    use_spacing = args.use_spacing.lower() == "true"
    spacing_size = tuple(map(float, args.spacing_size.split(',')))

    print(f"Testing transforms with resize size: {resize_size}")
    print(f"Using spacing transform: {use_spacing}")
    if use_spacing:
        print(f"Spacing size: {spacing_size}")
    print(f"CSV path: {args.csv_path}")
    print(f"Image directory: {args.img_dir}")

    # Create transforms for testing
    test_transforms = get_transforms(
        mode="val",
        resize_size=resize_size,
        resize_mode="trilinear",
        use_spacing=use_spacing,
        spacing_size=spacing_size
    )

    # Create a dataset without transforms first
    try:
        dataset = ADNIDataset(args.csv_path, args.img_dir)
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
                print(f"Transformed image: Tensor with shape {transformed_image.shape}")

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


if __name__ == "__main__":
    import sys

    # Remove "test_transforms" from sys.argv if it's the first argument
    if len(sys.argv) > 1 and sys.argv[1] == "test_transforms":
        # Save original command for error messages
        original_command = " ".join(sys.argv)
        # Remove the test_transforms argument before parsing other arguments
        sys.argv.pop(1)
        test_transforms()
    else:
        test_image_path_mapping()
