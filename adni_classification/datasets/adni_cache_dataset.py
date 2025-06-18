"""Dataset module for ADNI classification."""

import monai
from monai.data import CacheDataset
from typing import Optional, List, Dict, Any, Union

from adni_classification.datasets.adni_base_dataset import ADNIBaseDataset


class ADNICacheDataset(CacheDataset):
    """Dataset for ADNI MRI classification.

    This dataset loads 3D MRI images from the ADNI dataset and their corresponding labels.
    It uses MONAI's CacheDataset to cache data in memory for faster access.

    See ADNIBaseDataset for more details on supported CSV formats and classification modes.
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[monai.transforms.Compose] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        cache_num: Optional[int] = None,
        classification_mode: str = "CN_MCI_AD",
        mci_subtype_filter: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize the dataset.

        Args:
            csv_path: Path to the CSV file containing image metadata and labels
            img_dir: Path to the directory containing the image files
            transform: Optional transform to apply to the images
            cache_rate: The percentage of data to be cached (default: 1.0 = 100%)
            num_workers: Number of subprocesses to use for data loading (default: 0)
            cache_num: Number of items to cache. Default: None (cache_rate * len(data))
            classification_mode: Mode for classification, either "CN_MCI_AD" (3 classes) or "CN_AD" (2 classes)
            mci_subtype_filter: Optional filter for MCI subtypes in CN_AD mode.
                               Can be a single subtype (str) or list of subtypes (List[str]).
                               Valid subtypes: "SMC", "EMCI", "LMCI". Use None to include all MCI.
                               Examples: "EMCI", ["EMCI", "LMCI"], or None
        """
        # Initialize the base class to handle common functionality
        self.base = ADNIBaseDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            classification_mode=classification_mode,
            mci_subtype_filter=mci_subtype_filter,
            verbose=True
        )

        # Access the prepared data from the base class
        data_list = self.base.data_list

        # For convenience, expose common attributes from the base class
        self.csv_path = self.base.csv_path
        self.img_dir = self.base.img_dir
        self.classification_mode = self.base.classification_mode
        self.csv_format = self.base.csv_format
        self.label_map = self.base.label_map
        self.data = self.base.data
        self.image_paths = self.base.image_paths

        # Ensure there's enough cache for all data items to prevent index errors
        # If cache_num is not provided, use the total dataset size to ensure complete caching
        if cache_num is None:
            cache_num = len(data_list)
            print(f"Setting cache_num to dataset size: {cache_num}")

        # Initialize the CacheDataset
        super().__init__(
            data=data_list,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )


def test_image_path_mapping():
    """Test the image path mapping logic.

    This function creates a test dataset and prints information about the mapped image paths.
    It can be run directly to verify that the image path mapping logic is working correctly.
    """
    import argparse
    import torch
    import os
    import pandas as pd

    parser = argparse.ArgumentParser(description="Test the ADNI dataset image path mapping")
    parser.add_argument("--csv_path", type=str, default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_594images_client_all_train_475images.csv",
                        help="Path to the CSV file containing image metadata and labels")
    parser.add_argument("--img_dir", type=str, default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_611images_step3_skull_stripping",
                        help="Path to the directory containing the image files")
    parser.add_argument("--csv_format", type=str, choices=["original", "alternative"],
                        help="CSV format to test explicitly (will be auto-detected if not specified)")
    parser.add_argument("--cache_rate", type=float, default=1.0,
                        help="Percentage of data to cache (0.0-1.0)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for transforms (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--classification_mode", type=str, choices=["CN_MCI_AD", "CN_AD"], default="CN_MCI_AD",
                        help="Classification mode: 'CN_MCI_AD' for 3 classes or 'CN_AD' for 2 classes")
    args = parser.parse_args()

    # Parse device
    device = torch.device(args.device) if args.device else None

    print("Testing ADNI dataset image path mapping...")
    print(f"CSV path: {args.csv_path}")
    print(f"Image directory: {args.img_dir}")
    print(f"Cache rate: {args.cache_rate}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Device: {device}")
    print(f"Classification mode: {args.classification_mode}")

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
        dataset = ADNICacheDataset(
            args.csv_path,
            args.img_dir,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers,
            classification_mode=args.classification_mode,
        )

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_transforms":
        print("The test_transforms function has been moved to the transforms module.")
        print("Please use the following command instead:")
        print("python -m adni_classification.datasets.transforms [options]")
    else:
        test_image_path_mapping()
