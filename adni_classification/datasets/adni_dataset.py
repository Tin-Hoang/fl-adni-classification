"""Dataset module for ADNI classification using regular Dataset without caching."""

from typing import List, Optional, Union

import monai
from monai.data import Dataset

from adni_classification.datasets.adni_base_dataset import ADNIBaseDataset


class ADNIDataset(Dataset):
    """Dataset for ADNI MRI classification without caching.

    This dataset loads 3D MRI images from the ADNI dataset and their corresponding labels.
    Unlike the cached versions, this class uses regular Dataset which loads
    data on-the-fly without caching. This can be useful when memory is limited.

    See ADNIBaseDataset for more details on supported CSV formats and classification modes.
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[monai.transforms.Compose] = None,
        classification_mode: str = "CN_MCI_AD",
        mci_subtype_filter: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize the dataset.

        Args:
            csv_path: Path to the CSV file containing image metadata and labels
            img_dir: Path to the directory containing the image files
            transform: Optional transform to apply to the images
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
            verbose=True,
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

        # Initialize the MONAI Dataset with the prepared data list
        super().__init__(
            data=data_list,
            transform=transform,
        )


def test_normal_dataset():
    """Test the normal dataset without caching.

    This function creates a test dataset and prints information about the mapped image paths.
    It can be run directly to verify that the dataset works correctly.
    """
    import argparse
    import os

    import pandas as pd
    import torch

    parser = argparse.ArgumentParser(description="Test the ADNI normal dataset")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_594images_client_all_train_475images.csv",
        help="Path to the CSV file containing image metadata and labels",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_611images_step3_skull_stripping",
        help="Path to the directory containing the image files",
    )
    parser.add_argument(
        "--csv_format",
        type=str,
        choices=["original", "alternative"],
        help="CSV format to test explicitly (will be auto-detected if not specified)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use for transforms (e.g., 'cuda' or 'cpu')")
    parser.add_argument(
        "--classification_mode",
        type=str,
        choices=["CN_MCI_AD", "CN_AD"],
        default="CN_MCI_AD",
        help="Classification mode: 'CN_MCI_AD' for 3 classes or 'CN_AD' for 2 classes",
    )
    args = parser.parse_args()

    # Parse device
    device = torch.device(args.device) if args.device else None

    print("Testing ADNI normal dataset...")
    print(f"CSV path: {args.csv_path}")
    print(f"Image directory: {args.img_dir}")
    print(f"Device: {device}")
    print(f"Classification mode: {args.classification_mode}")

    # Read and print the first few rows of the CSV
    df = pd.read_csv(args.csv_path)
    print("\nFirst 10 rows of the CSV file:")
    print(df.head(10))

    # Print ID columns specifically for clarity
    if "Image Data ID" in df.columns:
        print("\nFirst 10 Image Data IDs in CSV:")
        for i, img_id in enumerate(df["Image Data ID"].head(10)):
            print(f"{i + 1}. {img_id}")

    if "image_id" in df.columns:
        print("\nFirst 10 image_ids in CSV:")
        for i, img_id in enumerate(df["image_id"].head(10)):
            print(f"{i + 1}. {img_id}")

    try:
        # Create the dataset without transforms first to see raw data
        print("\nAttempting to create normal dataset...")
        dataset = ADNIDataset(args.csv_path, args.img_dir, classification_mode=args.classification_mode)

        # If successful, print information about it
        print(f"Successfully created dataset with {len(dataset)} samples")
        print(f"Detected CSV format: {dataset.csv_format}")

        # Print information about the mapped image paths
        print("\nImage path mapping:")
        file_formats = {".nii": 0, ".nii.gz": 0}

        for image_id, file_path in list(dataset.image_paths.items())[:5]:  # Show first 5 mappings
            # Determine file format
            if file_path.endswith(".nii.gz"):
                file_formats[".nii.gz"] += 1
                format_str = "(.nii.gz)"
            elif file_path.endswith(".nii"):
                file_formats[".nii"] += 1
                format_str = "(.nii)"
            else:
                format_str = "(unknown format)"

            # Get the label for this image ID
            row = dataset.data[dataset.data["Image Data ID"] == image_id]
            if not row.empty:
                label_group = row["Group"].iloc[0]
                label_idx = dataset.label_map[label_group]
                print(
                    f"  {image_id} -> {os.path.basename(file_path)} "
                    f"{format_str} (Label: {label_group}, ID: {label_idx})"
                )
            else:
                print(f"  {image_id} -> {os.path.basename(file_path)} {format_str} (Label: unknown)")

        if len(dataset.image_paths) > 5:
            print(f"  ... and {len(dataset.image_paths) - 5} more")

        print("\nTest completed successfully.")

    except ValueError as e:
        print(f"\nError creating dataset: {e}")

    print("\nTest completed.")


if __name__ == "__main__":
    test_normal_dataset()
