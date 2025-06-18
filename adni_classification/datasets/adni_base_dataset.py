"""Base module for ADNI classification datasets.

This module provides a base class with common functionality for all ADNI dataset types.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import torch


class ADNIBaseDataset:
    """Base class for ADNI MRI classification datasets.

    This class provides shared functionality for all ADNI dataset classes including:
    - CSV format detection and standardization
    - Label mapping with classification mode support
    - Image file discovery and mapping
    - Data list creation

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

    Classification modes:
    - CN_MCI_AD: 3-class classification (CN=0, MCI=1, AD/Dementia=2)
    - CN_AD: 2-class classification (CN=0, AD/Dementia=1), where MCI samples are converted to AD
      In CN_AD mode, users can optionally filter MCI samples further by DX_bl column to include
      only specific MCI subtypes: SMC (Significant Memory Concern), EMCI (Early MCI), or LMCI (Late MCI)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        classification_mode: str = "CN_MCI_AD",
        mci_subtype_filter: Optional[Union[str, List[str]]] = None,
        verbose: bool = True
    ):
        """Initialize the base dataset.

        Args:
            csv_path: Path to the CSV file containing image metadata and labels
            img_dir: Path to the directory containing the image files
            classification_mode: Mode for classification, either "CN_MCI_AD" (3 classes) or "CN_AD" (2 classes)
            mci_subtype_filter: Optional filter for MCI subtypes in CN_AD mode.
                               Can be a single subtype (str) or list of subtypes (List[str]).
                               Valid subtypes: "SMC", "EMCI", "LMCI". Use None to include all MCI.
                               Examples: "EMCI", ["EMCI", "LMCI"], or None
            verbose: Whether to print detailed information during initialization
        """
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.classification_mode = classification_mode
        self.mci_subtype_filter = mci_subtype_filter
        self.verbose = verbose

        # Validate mci_subtype_filter parameter
        valid_mci_subtypes = ["CN", "SMC", "EMCI", "LMCI", "AD"]
        if self.mci_subtype_filter is not None:
            # Normalize to list format for easier processing
            if isinstance(self.mci_subtype_filter, str):
                self.mci_subtype_filter = [self.mci_subtype_filter]
            elif isinstance(self.mci_subtype_filter, list):
                if len(self.mci_subtype_filter) == 0:
                    raise ValueError("mci_subtype_filter list cannot be empty")
            else:
                raise ValueError("mci_subtype_filter must be a single subtype (str) or a list of subtypes (List[str])")

            # Validate all subtypes
            for subtype in self.mci_subtype_filter:
                if subtype not in valid_mci_subtypes:
                    raise ValueError(f"Invalid subtype: {subtype}. Valid subtypes are: {valid_mci_subtypes}")

            # Check classification mode
            if self.classification_mode != "CN_AD":
                raise ValueError("mci_subtype_filter can only be used with classification_mode='CN_AD'")

        if self.verbose:
            print("="*80)
            print(f"Initializing dataset with CSV path: {csv_path} and image directory: {img_dir}")
            if self.mci_subtype_filter is not None:
                subtypes_str = ", ".join(self.mci_subtype_filter)
                print(f"MCI subtype filter enabled: Only including [{subtypes_str}] samples from MCI group")

        # Load the CSV file
        self.data = pd.read_csv(csv_path)

        # Detect CSV format
        self.csv_format = self._detect_csv_format()

        # Map diagnosis groups to numeric labels based on detected format and classification mode
        if self.classification_mode == "CN_AD":
            # Binary classification - CN=0, AD=1, MCI->AD=1
            self.label_map = {"CN": 0, "MCI": 1, "AD": 1, "Dementia": 1}
            if self.verbose:
                print(f"Using binary classification mode (CN_AD): CN=0, MCI->AD=1, AD/Dementia=1")
        else:  # Default: "CN_MCI_AD"
            # 3-class classification - CN=0, MCI=1, AD=2
            self.label_map = {"CN": 0, "MCI": 1, "AD": 2, "Dementia": 2}
            if self.verbose:
                print(f"Using 3-class classification mode (CN_MCI_AD): CN=0, MCI=1, AD/Dementia=2")

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

        # Create a list of data dictionaries for Dataset
        self.data_list = self._create_data_list()

        if self.verbose:
            print(f"Found {len(self.image_paths)} image files in {img_dir}")
            print(f"Final dataset size: {len(self.data)} samples")

            # Print first 5 images with their label groups
            print("First 5 images with label groups:")
            for i, (idx, row) in enumerate(self.data.head(5).iterrows()):
                image_id = row["Image Data ID"]
                group = row["Group"]
                label = self.label_map[group]
                file_path = self.image_paths[image_id]
                print(f"{i+1}. ID: {image_id}, Label: {group} ({label}), File: {os.path.basename(file_path)}")

    def _create_data_list(self) -> List[Dict[str, Any]]:
        """Create a list of data dictionaries for Dataset.

        Returns:
            List of dictionaries, each containing image path and label
        """
        data_list = []
        for _, row in self.data.iterrows():
            image_id = row["Image Data ID"]
            label = self.label_map[row["Group"]]
            image_path = self.image_paths[image_id]

            data_list.append({
                "image": image_path,
                "label": label,
            })
        return data_list

    def _detect_csv_format(self) -> str:
        """Detect the format of the CSV file.

        Returns:
            String indicating the detected format: "original" or "alternative"
        """
        if "DX" in self.data.columns and "image_id" in self.data.columns:
            if self.verbose:
                print("Detected ALTERNATIVE CSV format with:")
                print("- 'DX' column for diagnosis (Dementia, MCI, CN)")
                print("- 'image_id' column for image identifiers (without 'I' prefix)")
            return "alternative"
        elif "Group" in self.data.columns and "Image Data ID" in self.data.columns:
            if self.verbose:
                print("Detected ORIGINAL CSV format with:")
                print("- 'Group' column for diagnosis (AD, MCI, CN)")
                print("- 'Image Data ID' column for image identifiers (with 'I' prefix)")
            return "original"
        else:
            raise ValueError("Unknown CSV format. CSV must have either 'Group' and 'Image Data ID' columns (original format) or 'DX' and 'image_id' columns (alternative format).")

    def _standardize_data(self) -> None:
        """Standardize the data based on the detected CSV format."""
        if self.csv_format == "original":
            # Filter out rows with missing labels
            self.data = self.data[self.data["Group"].isin(["AD", "MCI", "CN"])]

            # Store original Group distribution for verbose output
            if self.verbose:
                original_group_counts = self.data["Group"].value_counts().sort_index()

            # Apply MCI subtype filtering for CN_AD mode if specified
            if self.classification_mode == "CN_AD" and self.mci_subtype_filter is not None:
                if "DX_bl" not in self.data.columns:
                    raise ValueError("DX_bl column is required for MCI subtype filtering but not found in the CSV file")

                # Keep all non-MCI samples and only MCI samples with the specified DX_bl values
                mci_mask = self.data["Group"] == "MCI"
                non_mci_data = self.data[~mci_mask]
                mci_data = self.data[mci_mask]

                # Filter MCI data by DX_bl column - support multiple subtypes
                filtered_mci_data = mci_data[mci_data["DX_bl"].isin(self.mci_subtype_filter)]

                # Combine filtered data
                self.data = pd.concat([non_mci_data, filtered_mci_data], ignore_index=True)

                if self.verbose:
                    # Show DX_bl distribution of all MCI samples before filtering
                    print("DX_bl distribution of MCI samples (before subtype filtering):")
                    mci_dx_bl_counts = mci_data["DX_bl"].value_counts().sort_index()
                    for dx_bl, count in mci_dx_bl_counts.items():
                        print(f"  {dx_bl}: {count}")

                    subtypes_str = ", ".join(self.mci_subtype_filter)
                    print(f"Applied MCI subtype filtering: {len(mci_data)} MCI samples -> {len(filtered_mci_data)} [{subtypes_str}] samples")
        else:  # alternative format
            # Map DX column values to standard Group values
            dx_to_group = {"Dementia": "AD", "MCI": "MCI", "CN": "CN"}

            # Filter out rows with missing or invalid labels
            self.data = self.data[self.data["DX"].isin(list(dx_to_group.keys()))]

            # Store original DX distribution for verbose output
            if self.verbose:
                original_dx_counts = self.data["DX"].value_counts().sort_index()

            # Apply MCI subtype filtering for CN_AD mode if specified
            if self.classification_mode == "CN_AD" and self.mci_subtype_filter is not None:
                if "DX_bl" not in self.data.columns:
                    raise ValueError("DX_bl column is required for MCI subtype filtering but not found in the CSV file")

                # Keep all non-MCI samples and only MCI samples with the specified DX_bl values
                mci_mask = self.data["DX"] == "MCI"
                non_mci_data = self.data[~mci_mask]
                mci_data = self.data[mci_mask]

                # Filter MCI data by DX_bl column - support multiple subtypes
                filtered_mci_data = mci_data[mci_data["DX_bl"].isin(self.mci_subtype_filter)]

                # Combine filtered data
                self.data = pd.concat([non_mci_data, filtered_mci_data], ignore_index=True)

                if self.verbose:
                    # Show DX_bl distribution of all MCI samples before filtering
                    print("DX_bl distribution of MCI samples (before subtype filtering):")
                    mci_dx_bl_counts = mci_data["DX_bl"].value_counts().sort_index()
                    for dx_bl, count in mci_dx_bl_counts.items():
                        print(f"  {dx_bl}: {count}")

                    subtypes_str = ", ".join(self.mci_subtype_filter)
                    print(f"Applied MCI subtype filtering: {len(mci_data)} MCI samples -> {len(filtered_mci_data)} [{subtypes_str}] samples")

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

        if self.verbose:
            # Print summary of the standardized data
            print(f"Total samples after standardization: {len(self.data)}")

            # Show original data distribution before filtering
            if self.csv_format == "original":
                print("Original Group distribution (before filtering):")
                for group, count in original_group_counts.items():
                    print(f"  {group}: {count}")
            else:  # alternative format
                print("Original DX distribution (before filtering):")
                for dx, count in original_dx_counts.items():
                    print(f"  {dx}: {count}")

            print("Final Group distribution (after filtering):")
            for group, count in self.data["Group"].value_counts().items():
                print(f"  {group}: {count}")

            # Print the actual class distribution after applying label_map for the selected classification mode
            class_counts = self.data["Group"].map(self.label_map).value_counts().sort_index()
            print(f"Label distribution for {self.classification_mode} mode:")
            for label, count in class_counts.items():
                class_name = "CN" if label == 0 else "AD" if (label == 1 and self.classification_mode == "CN_AD") else "MCI" if (label == 1 and self.classification_mode == "CN_MCI_AD") else "AD"
                print(f"  Class {label} ({class_name}): {count}")
                if self.classification_mode == "CN_AD" and label == 1 and self.mci_subtype_filter is not None:
                    subtypes_str = ", ".join(self.mci_subtype_filter)
                    print(f"    (AD samples include [{subtypes_str}] MCI subtypes)")

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
                    if len(found_ids) < 5:
                        found_ids.append((image_id, file_path, parent_dir))

                    # Add to the mapping - if both .nii and .nii.gz exist for same ID,
                    # prioritize .nii.gz as it's likely the more recent/compressed version
                    if image_id not in image_paths or file.endswith('.nii.gz'):
                        image_paths[image_id] = file_path

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
