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
)


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

    The image files are expected to be in NiFTI format (.nii) and organized in a directory structure
    where the Image ID appears both in the filename (as a suffix before .nii) and in the parent directory.
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
            self.data["Image Data ID"] = self.data["image_id"].apply(
                lambda x: f"I{x}" if not x.startswith('I') else x
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
        """Find all .nii files in the root directory and map them to Image IDs.

        Args:
            root_dir: Root directory to search for .nii files

        Returns:
            Dictionary mapping Image IDs to file paths
        """
        image_paths = {}
        found_ids = []  # For debugging

        # Walk through the directory tree
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.nii'):
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

                    # Add to the mapping
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


def get_transforms(mode: str = "train", resize_size: Tuple[int, int, int] = (160, 160, 160), resize_mode: str = "trilinear") -> monai.transforms.Compose:
    """Get transforms for training or validation.

    Args:
        mode: Either "train" or "val"
        resize_size: Tuple of (height, width, depth) for resizing
        resize_mode: Interpolation mode for resizing

    Returns:
        A Compose transform
    """
    common_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
        # More robust intensity scaling using percentiles
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0,  # Use percentiles instead of fixed values
            a_max=100.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(
            keys=["image"],
            spatial_size=resize_size,
            mode=resize_mode,
        ),
    ]

    if mode == "train":
        train_transforms = [
            # Stronger augmentation for small dataset
            RandAffined(
                keys=["image"],
                prob=0.8,  # Increased probability
                rotate_range=(0.1, 0.1, 0.1),  # Increased rotation
                scale_range=(0.2, 0.2, 0.2),  # Increased scaling
                mode=("bilinear"),
            ),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Added flip in another axis
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=[0, 1]),
            # Add noise augmentation
            monai.transforms.RandGaussianNoised(
                keys=["image"],
                prob=0.5,
                mean=0.0,
                std=0.1,
            ),
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
    parser.add_argument("--csv_path", type=str, default="data/ADNI/ADNI1_Complete_1Yr_3T/ADNI1_Complete_1Yr_3T_codetest.csv",
                        help="Path to the CSV file containing image metadata and labels")
    parser.add_argument("--img_dir", type=str, default="data/ADNI/ADNI1_Complete_1Yr_3T/ADNI",
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
        for image_id, file_path in list(dataset.image_paths.items())[:5]:  # Show first 5 mappings
            # Get the label for this image ID
            row = dataset.data[dataset.data["Image Data ID"] == image_id]
            if not row.empty:
                label_group = row["Group"].iloc[0]
                label_idx = dataset.label_map[label_group]
                print(f"  {image_id} -> {os.path.basename(file_path)} (Label: {label_group}, ID: {label_idx})")
            else:
                print(f"  {image_id} -> {os.path.basename(file_path)} (Label: unknown)")

        if len(dataset.image_paths) > 5:
            print(f"  ... and {len(dataset.image_paths) - 5} more")

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


if __name__ == "__main__":
    test_image_path_mapping()
