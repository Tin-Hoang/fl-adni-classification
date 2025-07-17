#!/usr/bin/env python3
"""Script to remove duplicate ADNI files based on specific criteria.

This script iterates through ADNI image files in a specified directory, groups them by subject ID,
and applies a duplicate removal logic. The process involves two main steps:

1.  **Timestamp Duplicates:** For a given subject and preprocessing description, if multiple files
    have different timestamps, only the file with the earliest timestamp is kept.
2.  **Preprocessing Duplicates:** After resolving timestamp duplicates, if both `_Scaled` and
    `_Scaled_2` versions of a file exist for the same subject, the `_Scaled_2` version is
    prioritized and kept, while the `_Scaled` version is removed.

Finally, the script cleans up by removing any directories that become empty after duplicate
files are deleted.
"""

import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ADNIFileParser:
    """Parser for ADNI file paths."""

    def __init__(self, filepath: str):
        """Initialize parser with filepath.

        Args:
            filepath: Full path to the ADNI file
        """
        self.filepath = filepath
        self.parts = self._parse_path()

    def _parse_path(self) -> Dict[str, str]:
        """Parse the filepath into its components.

        Returns:
            Dictionary containing path components:
            - subject_id: Subject ID
            - preprocess_desc: Preprocessing description
            - datetime: Timestamp
            - image_id: Image ID
            - filename: Full filename
        """
        path = Path(self.filepath)

        # Find the subject ID pattern in the path
        # Look for patterns like "XXX_S_XXXX" which is the subject ID format
        subject_pattern = r'(\d{3}_S_\d{4})'
        subject_match = re.search(subject_pattern, str(path))

        if not subject_match:
            raise ValueError(f"Could not find subject ID in path: {self.filepath}")

        # Get the subject ID and its position in the path
        subject_id = subject_match.group(1)
        subject_pos = subject_match.start(1)

        # Extract the path components after the subject ID
        path_after_subject = str(path)[subject_pos:]
        path_parts = path_after_subject.split('/')

        if len(path_parts) < 5:
            raise ValueError(f"Invalid path structure after subject ID: {path_after_subject}")

        # Extract components
        preprocess_desc = path_parts[1]
        datetime_str = path_parts[2]
        image_id = path_parts[3]
        filename = path_parts[4]

        return {
            'subject_id': subject_id,
            'preprocess_desc': preprocess_desc,
            'datetime': datetime_str,
            'image_id': image_id,
            'filename': filename
        }

    @property
    def subject_id(self) -> str:
        return self.parts['subject_id']

    @property
    def preprocess_desc(self) -> str:
        return self.parts['preprocess_desc']

    @property
    def datetime_str(self) -> str:
        return self.parts['datetime']

    @property
    def datetime_obj(self) -> datetime:
        return datetime.strptime(self.parts['datetime'], '%Y-%m-%d_%H_%M_%S.%f')

    @property
    def image_id(self) -> str:
        return self.parts['image_id']

    @property
    def is_scaled_2(self) -> bool:
        return self.preprocess_desc.endswith('_Scaled_2')

    @property
    def parent_dir(self) -> str:
        """Get the parent directory of the file."""
        return str(Path(self.filepath).parent)

class DuplicateRemover:
    """Handles removal of duplicate ADNI files."""

    def __init__(self, input_dir: str):
        """Initialize with input directory.

        Args:
            input_dir: Root directory containing ADNI files
        """
        self.input_dir = Path(input_dir)
        self.files_by_subject: Dict[str, List[ADNIFileParser]] = defaultdict(list)
        self.removed_files: Set[str] = set()

    def collect_files(self) -> None:
        """Collect and parse all files in the input directory."""
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    filepath = os.path.join(root, file)
                    try:
                        parser = ADNIFileParser(filepath)
                        self.files_by_subject[parser.subject_id].append(parser)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid file path: {e}")

    def _handle_timestamp_duplicates(self, files: List[ADNIFileParser]) -> List[ADNIFileParser]:
        """Handle duplicate files with different timestamps.

        Args:
            files: List of files for the same subject

        Returns:
            List of files to keep
        """
        if len(files) <= 1:
            return files

        # Group files by datetime
        datetime_groups = defaultdict(list)
        for file in files:
            datetime_groups[file.datetime_str].append(file)

        if len(datetime_groups) > 1:
            # Sort datetime strings and keep the earliest
            sorted_dates = sorted(datetime_groups.keys())
            for date in sorted_dates[1:]:
                for file in datetime_groups[date]:
                    logger.warning(f"Removing duplicate timestamp file: {file.filepath}")
                    try:
                        os.remove(file.filepath)
                        self.removed_files.add(file.filepath)
                    except OSError as e:
                        logger.error(f"Failed to remove file {file.filepath}: {e}")

            return datetime_groups[sorted_dates[0]]

        return files

    def _handle_preprocess_duplicates(self, files: List[ADNIFileParser]) -> List[ADNIFileParser]:
        """Handle duplicate files with different preprocessing descriptions.

        Args:
            files: List of files for the same subject

        Returns:
            List of files to keep
        """
        if len(files) <= 1:
            return files

        # Group files by preprocessing description
        preprocess_groups = defaultdict(list)
        for file in files:
            preprocess_groups[file.preprocess_desc].append(file)

        if len(preprocess_groups) > 1:
            # Check for _Scaled and _Scaled_2 pairs
            scaled_files = []
            scaled_2_files = []

            for desc, group in preprocess_groups.items():
                if desc.endswith('_Scaled_2'):
                    scaled_2_files.extend(group)
                elif desc.endswith('_Scaled'):
                    scaled_files.extend(group)

            # Remove _Scaled files if _Scaled_2 exists
            if scaled_2_files and scaled_files:
                for file in scaled_files:
                    logger.warning(f"Removing non-_Scaled_2 file: {file.filepath}")
                    try:
                        os.remove(file.filepath)
                        self.removed_files.add(file.filepath)
                    except OSError as e:
                        logger.error(f"Failed to remove file {file.filepath}: {e}")

                return scaled_2_files

            # If no _Scaled_2 files, keep all files
            return files

        return files

    def _remove_empty_dirs(self) -> None:
        """Remove empty directories after file removal."""
        # Get all parent directories of removed files
        parent_dirs = {Path(file).parent for file in self.removed_files}

        # Add all ancestor directories up to the subject ID folder
        all_dirs_to_check = set()
        for dir_path in parent_dirs:
            current = dir_path
            while current != self.input_dir and current.parent != self.input_dir:
                all_dirs_to_check.add(current)
                current = current.parent

        # Sort directories by depth (deepest first) to ensure we remove from bottom up
        sorted_dirs = sorted(all_dirs_to_check, key=lambda p: len(p.parts), reverse=True)

        for dir_path in sorted_dirs:
            try:
                # Check if directory is empty
                if dir_path.exists() and not any(dir_path.iterdir()):
                    logger.info(f"Removing empty directory: {dir_path}")
                    dir_path.rmdir()
            except OSError as e:
                logger.error(f"Failed to remove directory {dir_path}: {e}")

    def process_duplicates(self) -> None:
        """Process and remove duplicate files."""
        for subject_id, files in self.files_by_subject.items():
            if len(files) > 2:
                logger.warning(
                    f"Subject {subject_id} has {len(files)} files. "
                    "This is more than expected. Please verify manually."
                )

            # New logic: Remove SURVEY folders
            files_to_keep_after_survey = []
            for file in files:
                if "SURVEY" in file.preprocess_desc:
                    logger.warning(f"Removing SURVEY file: {file.filepath}")
                    try:
                        os.remove(file.filepath)
                        self.removed_files.add(file.filepath)
                    except OSError as e:
                        logger.error(f"Failed to remove file {file.filepath}: {e}")
                else:
                    files_to_keep_after_survey.append(file)

            files = files_to_keep_after_survey

            # New logic: Remove Axial_Field_Mapping folders (including those with suffixes)
            files_to_keep_after_axial_field = []
            for file in files:
                if file.preprocess_desc.startswith("Axial_Field_Mapping"):
                    logger.warning(f"Removing Axial_Field_Mapping file: {file.filepath}")
                    try:
                        os.remove(file.filepath)
                        self.removed_files.add(file.filepath)
                    except OSError as e:
                        logger.error(f"Failed to remove file {file.filepath}: {e}")
                else:
                    files_to_keep_after_axial_field.append(file)

            files = files_to_keep_after_axial_field

            # New logic: Remove subfolders containing "AAHead_"
            files_to_keep_after_aahead = []
            for file in files:
                if "AAHead_" in file.preprocess_desc:
                    logger.warning(f"Removing file containing AAHead_ in path: {file.filepath}")
                    try:
                        os.remove(file.filepath)
                        self.removed_files.add(file.filepath)
                    except OSError as e:
                        logger.error(f"Failed to remove file {file.filepath}: {e}")
                else:
                    files_to_keep_after_aahead.append(file)

            files = files_to_keep_after_aahead

            # New logic: Remove Calibration_Scan and Cal_8HRBRAIN if other subfolders exist
            other_files_exist = any(file.preprocess_desc not in ["Calibration_Scan", "Cal_8HRBRAIN"] for file in files)
            files_to_keep_after_cal = []
            if other_files_exist:
                for file in files:
                    if file.preprocess_desc in ["Calibration_Scan", "Cal_8HRBRAIN"]:
                        logger.warning(f"Removing {file.preprocess_desc} file because other "
                                       f"subfolders exist: {file.filepath}")
                        try:
                            os.remove(file.filepath)
                            self.removed_files.add(file.filepath)
                        except OSError as e:
                            logger.error(f"Failed to remove file {file.filepath}: {e}")
                    else:
                        files_to_keep_after_cal.append(file)
                files = files_to_keep_after_cal

            # New logic: Remove Accelerated_Sagittal_MPRAGE_ND if Accelerated_Sagittal_MPRAGE exists
            accelerated_sagittal_mprage_exists = any(
                file.preprocess_desc == "Accelerated_Sagittal_MPRAGE" for file in files
            )
            files_to_keep_after_nd = []
            if accelerated_sagittal_mprage_exists:
                for file in files:
                    if file.preprocess_desc == "Accelerated_Sagittal_MPRAGE_ND":
                        logger.warning(f"Removing Accelerated_Sagittal_MPRAGE_ND file because "
                                       f"Accelerated_Sagittal_MPRAGE exists: {file.filepath}")
                        try:
                            os.remove(file.filepath)
                            self.removed_files.add(file.filepath)
                        except OSError as e:
                            logger.error(f"Failed to remove file {file.filepath}: {e}")
                    else:
                        files_to_keep_after_nd.append(file)
                files = files_to_keep_after_nd

            # Existing logic: Prioritize Accelerated_Sagittal_MPRAGE over REPE/SENS
            # This logic was already there, but we might need to re-evaluate if
            # accelerated_sagittal_mprage_exists variable name is clear enough now.
            # Keeping it as is for now, as it checks for the presence of the desired type
            # among the current set of files.
            accelerated_sagittal_mprage_exists_for_mprage_prioritization = any(
                file.preprocess_desc == "Accelerated_Sagittal_MPRAGE" for file in files
            )

            files_to_keep_after_prioritization = []
            if accelerated_sagittal_mprage_exists_for_mprage_prioritization:
                for file in files:
                    if file.preprocess_desc in ["MPRAGE_REPE", "MPRAGE_SENS"]:
                        logger.warning(f"Removing {file.preprocess_desc} file due to "
                                       f"Accelerated_Sagittal_MPRAGE priority: {file.filepath}")
                        try:
                            os.remove(file.filepath)
                            self.removed_files.add(file.filepath)
                        except OSError as e:
                            logger.error(f"Failed to remove file {file.filepath}: {e}")
                    else:
                        files_to_keep_after_prioritization.append(file)
                files = files_to_keep_after_prioritization

            # First handle timestamp duplicates
            files = self._handle_timestamp_duplicates(files)

            # Then handle preprocessing description duplicates
            files = self._handle_preprocess_duplicates(files)

            # Check if only one subfolder remains after processing
            if files:
                remaining_parent_dirs = set(file.parent_dir for file in files)
                if len(remaining_parent_dirs) > 1:
                    logger.warning(
                        f"Subject {subject_id} still contains {len(remaining_parent_dirs)} "
                        f"subfolders after processing: {list(remaining_parent_dirs)}. "
                        "Please verify manually."
                    )

            if len(files) > 1:
                logger.warning(
                    f"Subject {subject_id} still has {len(files)} files after processing. "
                    "Please verify manually."
                )

        # Remove empty directories after all file removals
        self._remove_empty_dirs()

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Remove duplicate ADNI files")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing ADNI files"
    )

    args = parser.parse_args()

    remover = DuplicateRemover(args.input_dir)
    remover.collect_files()
    remover.process_duplicates()

    logger.info("Duplicate removal completed")

if __name__ == "__main__":
    main()
