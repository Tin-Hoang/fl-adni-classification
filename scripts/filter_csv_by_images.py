#!/usr/bin/env python3
"""
This script filters a given CSV file based on the presence of corresponding image files
in a specified directory. It is designed for datasets where metadata is stored in a CSV
and the actual data (images) are stored in a hierarchical directory structure.

This script is specifically designed for the ADNI dataset in the context of the
federated learning classification project (`fl-adni-classification`). It processes
ADNI imaging data, typically stored in `.nii` or `.nii.gz` format within participant-specific
subdirectories, and filters a corresponding clinical or imaging metadata CSV file.
The goal is to ensure that only participants and imaging sessions for whom `.nii` or
`.nii.gz` files are actually present in the specified image directory are included in the
filtered CSV output. This is crucial for maintaining data integrity and consistency
between the metadata used for model training and the available image files.

The script performs the following steps:
1. Scans a specified image directory to find all .nii and .nii.gz image files.
2. Extracts subject and image IDs from the file paths of the discovered images.
   Multiple parsing patterns are attempted to handle variations in file path structures.
3. Reads the input CSV file, identifying the columns corresponding to subject and image IDs.
4. Filters the rows in the CSV to keep only those where the subject and image ID pair
   matches a pair found during the image file scanning process.
   Normalization (like stripping leading zeros) is applied to image IDs for better matching.
   A fallback to subject ID only matching is attempted if image ID matching yields no results.
5. Writes the filtered data to a new CSV file.

This script is particularly useful for ensuring that the metadata used for analysis
corresponds exactly to the available image data, preventing errors due to missing files.
It includes detailed logging to help track the process and identify potential issues.

Usage:
    python filter_csv_by_images.py <image_directory> <input_csv_file> \
        [--output <output_csv_file>] [--subject-id-col <col_name>] [--image-id-col <col_name>] \
        [--log-level <level>]

Arguments:
    image_directory: Path to the directory containing .nii or .nii.gz image files.
    input_csv_file: Path to the input CSV file to be filtered.

Options:
    --output: Path for the filtered output CSV file. If not provided,
              a filename will be generated based on the input CSV.
    --subject-id-col: Name of the column in the CSV containing subject IDs (default: "subject_id").
    --image-id-col: Name of the column in the CSV containing image IDs (default: "image_id").
    --log-level: Set the logging level (DEBUG, INFO, WARNING, ERROR) (default: DEBUG).

Examples:
    # Filter a CSV using images in a directory, generating output filename:
    python filter_csv_by_images.py /path/to/images data.csv

    # Filter a CSV and specify output filename and custom column names:
    python filter_csv_by_images.py /path/to/images data.csv --output filtered_data.csv \
        --subject-id-col Subject --image-id-col ImageID
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageFileParser:
    """Parser for image file paths."""

    def __init__(self, filepath: str):
        """Initialize parser with filepath.

        Args:
            filepath: Full path to the image file
        """
        self.filepath = filepath
        self.subject_id, self.image_id = self._parse_path()

    def _parse_path(self) -> Tuple[str, str]:
        """Parse the filepath to extract subject_id and image_id.

        Returns:
            Tuple containing (subject_id, image_id)
        """
        path = Path(self.filepath)
        path_str = str(path)

        # Log the path we're trying to parse
        logger.debug(f"Parsing path: {path_str}")

        # Find the subject ID pattern in the path (format: XXX_S_XXXX)
        subject_pattern = r'(\d{3}_S_\d{4})'
        subject_match = re.search(subject_pattern, path_str)

        if not subject_match:
            raise ValueError(f"Could not find subject ID in path: {self.filepath}")

        subject_id = subject_match.group(1)
        logger.debug(f"Extracted subject_id: {subject_id}")

        # Try multiple patterns to find the image ID

        # 1. First look for a directory named IXXXXX
        dir_pattern = r'/I(\d+)/'
        dir_match = re.search(dir_pattern, path_str)
        if dir_match:
            image_id = dir_match.group(1)
            logger.debug(f"Found image_id in directory name: {image_id}")
            return subject_id, image_id

        # 2. Look for I or S followed by digits at the end of the filename
        filename = path.name
        filename_end_pattern = r'[IS](\d+)\.(nii|nii\.gz)'
        filename_end_match = re.search(filename_end_pattern, filename)
        if filename_end_match:
            image_id = filename_end_match.group(1)
            logger.debug(f"Found image_id at end of filename: {image_id}")
            return subject_id, image_id

        # 3. Look for any pattern like IXXXXX or I_XXXXX in the path
        image_id_pattern = r'I(?:_)?(\d+)'
        image_id_match = re.search(image_id_pattern, path_str)
        if image_id_match:
            image_id = image_id_match.group(1)
            logger.debug(f"Found image_id with general pattern: {image_id}")
            return subject_id, image_id

        # If still not found, try to find any sequence of digits after _I
        fallback_pattern = r'_I(\d+)'
        fallback_match = re.search(fallback_pattern, path_str)
        if fallback_match:
            image_id = fallback_match.group(1)
            logger.debug(f"Found image_id with fallback pattern: {image_id}")
            return subject_id, image_id

        # Last resort - try to extract any digit sequence that might be an ID
        # Look for digit sequences of 5 or more digits that are likely to be IDs
        last_resort_pattern = r'(\d{5,})'
        last_resort_matches = re.findall(last_resort_pattern, path_str)
        if last_resort_matches:
            # Use the first match as image ID
            image_id = last_resort_matches[0]
            logger.debug(f"Found potential image_id as numeric sequence: {image_id}")
            return subject_id, image_id

        raise ValueError(f"Could not find image ID in path: {self.filepath}")


class CSVFilter:
    """Filters CSV file based on existing images."""

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        output_path: str,
        subject_id_col: str,
        image_id_col: str
    ):
        """Initialize with input parameters.

        Args:
            image_dir: Directory containing .nii or .nii.gz files
            csv_path: Path to the CSV file to filter
            output_path: Path for the filtered output CSV
            subject_id_col: Column name for subject ID in the CSV
            image_id_col: Column name for image ID in the CSV
        """
        self.image_dir = Path(image_dir)
        self.csv_path = csv_path
        self.output_path = output_path
        self.subject_id_col = subject_id_col
        self.image_id_col = image_id_col
        self.subject_image_pairs = set()
        self.image_ids = set()
        self.subject_by_image = {}

    def collect_images(self) -> None:
        """Collect all .nii and .nii.gz files in the input directory and extract subject/image IDs."""
        logger.info(f"Collecting .nii and .nii.gz files from: {self.image_dir}")
        count = 0
        processed_image_ids = []

        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(('.nii', '.nii.gz')):
                    filepath = os.path.join(root, file)
                    try:
                        parser = ImageFileParser(filepath)
                        self.subject_image_pairs.add((parser.subject_id, parser.image_id))
                        self.image_ids.add(parser.image_id)
                        self.subject_by_image[parser.image_id] = parser.subject_id
                        count += 1

                        # Collect a sample of processed image IDs for inspection
                        if len(processed_image_ids) < 10:
                            processed_image_ids.append(parser.image_id)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid file path: {e}")

        logger.info(f"Found {count} .nii and .nii.gz files with valid subject and image IDs")
        logger.info(f"Unique subject-image pairs: {len(self.subject_image_pairs)}")
        logger.info(f"Unique image IDs: {len(self.image_ids)}")
        logger.info(f"First 10 image IDs: {processed_image_ids[:10]}")

    def filter_csv(self) -> None:
        """Filter the CSV file based on the collected subject-image pairs."""
        logger.info(f"Reading CSV file: {self.csv_path}")
        df = pd.read_csv(self.csv_path, low_memory=False)

        if self.subject_id_col not in df.columns:
            raise ValueError(f"Subject ID column '{self.subject_id_col}' not found in CSV")

        if self.image_id_col not in df.columns:
            raise ValueError(f"Image ID column '{self.image_id_col}' not found in CSV")

        # Print sample values for debugging
        logger.info(f"CSV column names: {list(df.columns)}")
        logger.info(f"CSV sample subject IDs: {df[self.subject_id_col].head(10).tolist()}")
        logger.info(f"CSV sample image IDs: {df[self.image_id_col].head(10).tolist()}")
        logger.info(f"Sample image IDs from files: {list(self.image_ids)[:10]}")

        # Check for data type issues and convert image ID column to string
        logger.info(f"CSV image ID column dtype: {df[self.image_id_col].dtype}")

        # Create normalized versions of the relevant columns for matching
        df['image_id_norm'] = df[self.image_id_col].astype(str).str.strip().str.lstrip('0')
        df['subject_id_norm'] = df[self.subject_id_col].astype(str).str.strip()

        # Print first few rows with normalized columns to verify transformation
        logger.info("First 10 rows with normalized columns:")
        logger.info(df[[self.image_id_col, 'image_id_norm', self.subject_id_col, 'subject_id_norm']].head(10).to_string())

        # Print normalized image IDs from files for comparison
        norm_file_image_ids = {img_id.lstrip('0') for img_id in self.image_ids}
        logger.info(f"First 10 normalized image IDs from files: {list(norm_file_image_ids)[:10]}")

        # Check for any potential matches by cross-checking
        csv_image_ids = set(df['image_id_norm'].tolist())
        common_ids = norm_file_image_ids.intersection(csv_image_ids)
        logger.info(f"Common image IDs (after normalization): {len(common_ids)}")
        if common_ids:
            logger.info(f"Sample common IDs: {list(common_ids)[:10]}")
        else:
            logger.info("No common IDs found after normalization")

            # Try a more relaxed matching approach by checking if image IDs from files
            # are contained within any CSV image ID strings (partial matching)
            partial_matches = []
            for file_id in list(norm_file_image_ids)[:20]:  # Check first 20 file IDs
                for csv_id in list(csv_image_ids)[:1000]:  # Against first 1000 CSV IDs
                    if file_id in csv_id or csv_id in file_id:
                        partial_matches.append((file_id, csv_id))
                        if len(partial_matches) >= 10:
                            break
                if len(partial_matches) >= 10:
                    break

            if partial_matches:
                logger.info(f"Found some partial matches: {partial_matches}")
            else:
                logger.info("No partial matches found either")

        # Primary filtering based on image ID
        filtered_df = df[df['image_id_norm'].isin(norm_file_image_ids)]
        logger.info(f"After image ID filtering: {len(filtered_df)} rows")

        # If no matches found, try a more flexible approach
        if len(filtered_df) == 0:
            logger.info("Trying more flexible matching...")

            # Try matching with both image ID and subject ID as alternatives
            subject_matches = df[df['subject_id_norm'].isin([s for s, _ in self.subject_image_pairs])]
            logger.info(f"Rows matching by subject ID only: {len(subject_matches)}")

            if len(subject_matches) > 0:
                filtered_df = subject_matches
                logger.info("Using subject ID matches instead of image ID matches")

        # Remove temporary columns
        if len(filtered_df) > 0:
            filtered_df = filtered_df.drop(['image_id_norm', 'subject_id_norm'], axis=1)

        logger.info(f"Final filtered CSV contains {len(filtered_df)} rows")

        # Write the filtered dataframe to the output file
        filtered_df.to_csv(self.output_path, index=False)
        logger.info(f"Filtered CSV written to: {self.output_path}")


def generate_output_filename(csv_path: str) -> str:
    """Generate an output filename based on the input CSV filename.

    Args:
        csv_path: Path to the input CSV file

    Returns:
        Generated output filename
    """
    csv_file = Path(csv_path)

    # Extract stem (filename without extension) from the file
    csv_stem = csv_file.stem

    # Use the same directory as the input file
    output_dir = csv_file.parent

    # Create new filename with _filtered suffix
    output_filename = f"{csv_stem}_filtered.csv"

    return str(output_dir / output_filename)


def main():
    """Main function to run the CSV filter."""
    parser = argparse.ArgumentParser(description="Filter a CSV file based on existing images in a directory")

    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing .nii or .nii.gz files"
    )

    parser.add_argument(
        "csv_file",
        type=str,
        default="data/ADNI/All_Subjects_MRI_Images_11Apr2025_ADNIMERGE_11Apr2025_merged.csv",
        help="Path to the CSV file to filter"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path for the filtered output CSV file (optional, will generate if not provided)"
    )

    parser.add_argument(
        "--subject-id-col",
        type=str,
        default="subject_id",
        help="Column name for subject ID in the CSV"
    )

    parser.add_argument(
        "--image-id-col",
        type=str,
        default="image_id",
        help="Column name for image ID in the CSV"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="Set the logging level"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate input paths exist
    if not os.path.exists(args.image_dir):
        logger.error(f"Image directory not found: {args.image_dir}")
        return 1

    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return 1

    # Generate output filename if not provided
    output_path = args.output
    if not output_path:
        output_path = generate_output_filename(args.csv_file)
        logger.info(f"Output filename automatically generated: {output_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize and run the CSV filter
    csv_filter = CSVFilter(
        args.image_dir,
        args.csv_file,
        output_path,
        args.subject_id_col,
        args.image_id_col
    )

    try:
        csv_filter.collect_images()
        csv_filter.filter_csv()
        logger.info("CSV filtering completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during CSV filtering: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
