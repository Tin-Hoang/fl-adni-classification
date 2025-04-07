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
    The dataset expects a CSV file with the following columns:
    - Image Data ID: The ID of the image in the ADNI database
    - Subject: The subject ID (e.g., "136_S_1227")
    - Group: The diagnosis group (AD, MCI, NC)
    - Description: The image description (e.g., "MPR; ; N3; Scaled")
    
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
        
        # Map diagnosis groups to numeric labels
        self.label_map = {"AD": 0, "MCI": 1, "CN": 2}
        
        # Filter out rows with missing labels
        self.data = self.data[self.data["Group"].isin(self.label_map.keys())]
        
        # Create a mapping from Image Data ID to file path
        self.image_paths = self._find_image_files(img_dir)
        
        # Filter out rows with missing image files
        self.data = self.data[self.data["Image Data ID"].isin(self.image_paths.keys())]
        
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        print(f"Found {len(self.image_paths)} image files in {img_dir}")
        
        # Print class distribution
        class_counts = self.data["Group"].value_counts()
        print("Class distribution:")
        for group, count in class_counts.items():
            print(f"  {group}: {count}")
    
    def _find_image_files(self, root_dir: str) -> Dict[str, str]:
        """Find all .nii files in the root directory and map them to Image IDs.
        
        Args:
            root_dir: Root directory to search for .nii files
            
        Returns:
            Dictionary mapping Image IDs to file paths
        """
        image_paths = {}
        
        # Walk through the directory tree
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.nii'):
                    file_path = os.path.join(root, file)
                    
                    # Extract the Image ID from the filename (suffix before .nii)
                    # Example: ADNI_002_S_0413_MR_MPR____N3__Scaled_Br_20070216232854688_S14782_I40657.nii
                    parts = file.split('_')
                    for part in reversed(parts):
                        if part.startswith('I') and part[1:].isdigit():
                            image_id = part
                            break
                    else:
                        # If no Image ID found in filename, try to extract from parent directory
                        parent_dir = os.path.basename(root)
                        if parent_dir.startswith('I') and parent_dir[1:].isdigit():
                            image_id = parent_dir
                        else:
                            # Skip this file if no Image ID found
                            continue
                    
                    # Verify that the Image ID in the parent directory matches the one in the filename
                    parent_dir = os.path.basename(root)
                    if parent_dir.startswith('I') and parent_dir[1:].isdigit() and parent_dir != image_id:
                        print(f"Warning: Image ID mismatch - filename: {image_id}, parent dir: {parent_dir}")
                        continue
                    
                    # Add to the mapping
                    image_paths[image_id] = file_path
        
        return image_paths

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
    if mode == "train":
        return Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1000,
                    a_max=1000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resized(
                    keys=["image"],
                    spatial_size=resize_size,
                    mode=resize_mode,
                ),
                RandAffined(
                    keys=["image"],
                    prob=0.5,
                    rotate_range=(0.05, 0.05, 0.05),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear"),
                ),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandRotate90d(keys=["image"], prob=0.5, spatial_axes=[0, 1]),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:  # val
        return Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1000,
                    a_max=1000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resized(
                    keys=["image"],
                    spatial_size=resize_size,
                    mode=resize_mode,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )


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
    args = parser.parse_args()
    
    print("Testing ADNI dataset image path mapping...")
    print(f"CSV path: {args.csv_path}")
    print(f"Image directory: {args.img_dir}")
    
    # Create a dataset without transforms
    dataset = ADNIDataset(args.csv_path, args.img_dir)
    
    # Print information about the mapped image paths
    print("\nImage path mapping:")
    for image_id, file_path in list(dataset.image_paths.items())[:5]:  # Show first 5 mappings
        print(f"  {image_id} -> {file_path}")
    
    if len(dataset.image_paths) > 5:
        print(f"  ... and {len(dataset.image_paths) - 5} more")
    
    # Check if all Image IDs in the CSV are mapped to file paths
    csv_image_ids = set(dataset.data["Image Data ID"].unique())
    mapped_image_ids = set(dataset.image_paths.keys())
    
    missing_ids = csv_image_ids - mapped_image_ids
    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} Image IDs in the CSV are not mapped to file paths:")
        for image_id in list(missing_ids)[:5]:  # Show first 5 missing IDs
            print(f"  {image_id}")
        if len(missing_ids) > 5:
            print(f"  ... and {len(missing_ids) - 5} more")
    else:
        print("\nAll Image IDs in the CSV are mapped to file paths.")
    
    # Check if all mapped Image IDs are in the CSV
    extra_ids = mapped_image_ids - csv_image_ids
    if extra_ids:
        print(f"\nNote: {len(extra_ids)} mapped Image IDs are not in the CSV:")
        for image_id in list(extra_ids)[:5]:  # Show first 5 extra IDs
            print(f"  {image_id}")
        if len(extra_ids) > 5:
            print(f"  ... and {len(extra_ids) - 5} more")
    else:
        print("\nAll mapped Image IDs are in the CSV.")
    
    print("\nTest completed.")


if __name__ == "__main__":
    test_image_path_mapping()