#!/usr/bin/env python3
"""
ADNI MRI Preprocessing Script

This script performs standard preprocessing steps on ADNI MRI images in the following order:
1. Skull Stripping
2. Registration to a standard template
3. Resampling to 1mm isotropic spacing

Usage:
    python preprocess_mri.py --input input_folder [--output output_dir] [--template template.nii.gz]

Requirements:
    - ANTs (for registration and resampling)
    - FSL (for BET skull stripping)
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from typing import List, Optional

from tqdm import tqdm


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ADNI MRI images.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing MRI image files (.nii or .nii.gz)")
    parser.add_argument("--output", type=str, default=None,
                        help="Base output directory (default: parent dir of input directory)")
    parser.add_argument(
        "--template",
        type=str,
        default="data/ICBM152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii.gz",
        help="Template for registration"
    )
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars")
    parser.add_argument(
        "--include-dirs-regex",
        type=str,
        default=None,
        help=(
            "Comma-separated list of regex patterns to include specific first-level "
            "subdirectories. Example: '34.*,94*'"
        )
    )

    args = parser.parse_args()

    # If output is not specified, use the parent directory of the input directory
    if args.output is None:
        input_dir_name = os.path.basename(os.path.normpath(args.input))
        args.output = os.path.join(os.path.dirname(os.path.normpath(args.input)), input_dir_name)
        logging.info(f"Output directory not specified. Using: {args.output}")

    return args


def find_nifti_files(input_dir: str, include_dirs_regex: Optional[List[str]] = None) -> List[str]:
    """
    Find all NIFTI files (.nii or .nii.gz) in the input directory and subdirectories,
    optionally filtering first-level subdirectories by regex patterns.

    Args:
        input_dir: Directory to search for NIFTI files
        include_dirs_regex: Optional list of regex patterns to filter first-level directories

    Returns:
        List of paths to NIFTI files
    """
    all_nifti_files: List[str] = []

    # List first-level directories
    try:
        first_level_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    except FileNotFoundError:
        logging.error(f"Input directory not found: {input_dir}")
        return []

    # Filter directories based on regex if provided
    filtered_dirs = []
    if include_dirs_regex:
        logging.info(f"Filtering first-level directories using patterns: {include_dirs_regex}")
        import re
        for dir_name in first_level_dirs:
            if any(re.fullmatch(pattern.strip(), dir_name) for pattern in include_dirs_regex):
                filtered_dirs.append(dir_name)
                logging.debug(f"Matched directory: {dir_name}")
            else:
                logging.debug(f"Skipping directory (no match): {dir_name}")
        if not filtered_dirs:
             logging.warning(f"No directories matched the provided regex patterns: {include_dirs_regex}")
             return []
    else:
        filtered_dirs = first_level_dirs
        logging.info("No directory regex patterns provided. Including all first-level directories.")


    # Search for NIFTI files within the filtered directories
    for dir_name in filtered_dirs:
        subdir_path = os.path.join(input_dir, dir_name)
        # Use glob with recursive=True to find all .nii and .nii.gz files in all subdirectories
        nii_files = glob.glob(os.path.join(subdir_path, "**", "*.nii"), recursive=True)
        nii_gz_files = glob.glob(os.path.join(subdir_path, "**", "*.nii.gz"), recursive=True)
        all_nifti_files.extend(nii_files)
        all_nifti_files.extend(nii_gz_files)

    return all_nifti_files


def check_dependencies() -> None:
    """Check if required external dependencies are installed."""
    tools = {
        "ResampleImageBySpacing": "ANTs",
        "antsRegistrationSyN.sh": "ANTs",
        "bet": "FSL"
    }

    missing_tools = []

    for tool, package in tools.items():
        try:
            subprocess.run(["which", tool], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            missing_tools.append(f"{tool} (part of {package})")

    if missing_tools:
        raise PreprocessingError(f"Missing required tools: {', '.join(missing_tools)}")


def run_command(command: list) -> str:
    """
    Execute a shell command and return result.

    Args:
        command: List containing the command and its arguments

    Returns:
        Command output

    Raises:
        PreprocessingError: If command execution fails
    """
    try:
        logging.info(f"Executing: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing {command[0]}: {e.stderr}"
        logging.error(error_msg)
        raise PreprocessingError(error_msg) from e


def resample_image(input_path: str, output_path: str) -> None:
    """
    Resample MRI image to 1mm isotropic spacing.

    Args:
        input_path: Path to input MRI image
        output_path: Path to save resampled image

    Raises:
        PreprocessingError: If resampling fails
    """
    logging.info(f"Step 3: Resampling image {os.path.basename(input_path)} to 1mm isotropic spacing")
    command = ["ResampleImageBySpacing", "3", input_path, output_path, "1", "1", "1"]
    run_command(command)


def register_to_template(input_path: str, template_path: str, output_prefix: str, final_output_path: str) -> None:
    """
    Register MRI image to standard template.

    Args:
        input_path: Path to resampled MRI image
        template_path: Path to template image
        output_prefix: Prefix for intermediate output files
        final_output_path: Path to save the final registered image

    Raises:
        PreprocessingError: If registration or file copying fails
    """
    logging.info(f"Step 2: Registering {os.path.basename(input_path)} to standard template")
    command = [
        "antsRegistrationSyN.sh",
        "-d", "3",
        "-f", template_path,
        "-m", input_path,
        "-o", output_prefix,
        "-n", "28",            # Use 28 threads for parallel processing
        "-e", "42"             # Set random seed for reproducibility
    ]

    run_command(command)

    # The registration script produces a warped image with this naming convention
    warped_image = f"{output_prefix}Warped.nii.gz"

    # Copy the warped image to the final output path
    if os.path.exists(warped_image):
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        # Use copy command to maintain the original format
        copy_command = ["cp", warped_image, final_output_path]
        run_command(copy_command)
    else:
        error_msg = f"Expected warped image not found: {warped_image}"
        logging.error(error_msg)
        raise PreprocessingError(error_msg)


def skull_strip(input_path: str, output_path: str) -> None:
    """
    Remove non-brain tissue using BET.

    Args:
        input_path: Path to registered MRI image
        output_path: Path to save skull-stripped image

    Raises:
        PreprocessingError: If skull stripping fails
    """
    logging.info(f"Step 1: Skull stripping {os.path.basename(input_path)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = ["bet", input_path, output_path]
    run_command(command)


def create_step_directory(base_dir: str, step_number: int, step_name: str) -> str:
    """
    Create a directory for a specific preprocessing step.

    Args:
        base_dir: Base output directory
        step_number: Number of the processing step
        step_name: Name of the processing step

    Returns:
        Path to the created directory
    """
    # Get the parent directory and the base name of the input directory
    parent_dir = os.path.dirname(os.path.normpath(base_dir))
    base_name = os.path.basename(os.path.normpath(base_dir))

    # Create the step directory in the parent directory of the input
    step_dir = os.path.join(parent_dir, f"{base_name}_step{step_number}_{step_name.lower().replace(' ', '_')}")
    os.makedirs(step_dir, exist_ok=True)
    return step_dir


def get_relative_path(file_path: str, base_dir: str) -> str:
    """
    Get the relative path of a file from a base directory.
    Args:
        file_path: Absolute path to file
        base_dir: Base directory
    Returns:
        Relative path from base_dir to file_path
    """
    abs_base_dir = os.path.abspath(base_dir)
    abs_file_path = os.path.abspath(file_path)
    # Remove base_dir from the beginning of file_path to get the relative path
    if abs_file_path.startswith(abs_base_dir):
        rel_path = abs_file_path[len(abs_base_dir):].lstrip(os.sep)
        return rel_path
    else:
        # If file_path is not within base_dir, return just the filename
        return os.path.basename(file_path)


def preprocess_mri_file(
    input_file: str,
    input_dir: str,
    output_dir: str,
    template_file: str,
    show_progress: bool = True
) -> None:
    """
    Execute the full preprocessing pipeline for a single file.

    Args:
        input_file: Path to input MRI image
        input_dir: Base input directory to calculate relative path
        output_dir: Base directory to save output files
        template_file: Path to template image for registration
        show_progress: Whether to show progress bars

    Raises:
        PreprocessingError: If any preprocessing step fails
    """
    # Get the relative path from the input directory
    rel_path = get_relative_path(input_file, input_dir)

    # Create step-specific directories
    step1_dir = create_step_directory(output_dir, 1, "skull_stripping")
    step2_dir = create_step_directory(output_dir, 2, "registration")
    step3_dir = create_step_directory(output_dir, 3, "resampling")

    # Define output file paths for each step (maintaining subfolder structure)
    skull_stripped_path = os.path.join(step1_dir, rel_path)
    registered_path = os.path.join(step2_dir, rel_path)
    resampled_path = os.path.join(step3_dir, rel_path)

    # Create directories for the output files
    os.makedirs(os.path.dirname(resampled_path), exist_ok=True)
    os.makedirs(os.path.dirname(registered_path), exist_ok=True)
    os.makedirs(os.path.dirname(skull_stripped_path), exist_ok=True)

    # Define temporary directory for intermediate files
    temp_dir = os.path.join(os.path.dirname(output_dir), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Get the base filename without extension for temporary files
    input_basename = os.path.basename(input_file)
    if input_basename.endswith('.nii.gz'):
        input_basename = input_basename[:-7]
    elif input_basename.endswith('.nii'):
        input_basename = input_basename[:-4]

    # Define temporary file paths for intermediate processing
    basename_clean = os.path.basename(skull_stripped_path).replace('.nii.gz', '').replace('.nii', '')
    temp_registration_prefix = os.path.join(temp_dir, f"{basename_clean}_reg_")

    try:
        # Define preprocessing steps
        steps = [
            ("Skull Stripping", lambda: skull_strip(input_file, skull_stripped_path)),
            (
                "Registration",
                lambda: register_to_template(
                    skull_stripped_path, template_file, temp_registration_prefix, registered_path
                )
            ),
            ("Resampling", lambda: resample_image(registered_path, resampled_path))
        ]

        # Execute steps with progress bar if enabled
        if show_progress:
            for _step_name, step_func in tqdm(steps, desc=f"Processing {os.path.basename(input_file)}", leave=False):
                step_func()
        else:
            for step_name, step_func in steps:
                logging.info(f"  - Step: {step_name}")
                step_func()

        logging.info(f"Preprocessing complete for {rel_path}.")
        logging.info(f"Final output: {skull_stripped_path}")
    except Exception as e:
        logging.error(f"Failed preprocessing {rel_path}: {str(e)}")
        raise


def is_file_fully_processed(input_file: str, output_dir: str, input_dir: str) -> bool:
    """
    Check if a file has been fully processed by verifying all output files exist.

    Args:
        input_file: Path to input MRI image
        output_dir: Base output directory
        input_dir: Base input directory

    Returns:
        True if all output files exist, False otherwise
    """
    # Get the relative path from the input directory
    rel_path = get_relative_path(input_file, input_dir)

    # Create step-specific directories (based on the new order)
    step1_dir = create_step_directory(output_dir, 1, "skull_stripping")
    step2_dir = create_step_directory(output_dir, 2, "registration")
    step3_dir = create_step_directory(output_dir, 3, "resampling")

    # Get the base path without extension
    base_path = os.path.splitext(rel_path)[0]
    if base_path.endswith('.nii'):
        base_path = os.path.splitext(base_path)[0]

    # Check both .nii and .nii.gz extensions for each step's output
    skull_stripped_paths = [
        os.path.join(step1_dir, base_path + ext)
        for ext in ['.nii', '.nii.gz']
    ]
    registered_paths = [
        os.path.join(step2_dir, base_path + ext)
        for ext in ['.nii', '.nii.gz']
    ]
    resampled_paths = [
        os.path.join(step3_dir, base_path + ext)
        for ext in ['.nii', '.nii.gz']
    ]

    # Check if at least one version of each output file exists
    return (
        any(os.path.exists(path) for path in skull_stripped_paths) and
        any(os.path.exists(path) for path in registered_paths) and
        any(os.path.exists(path) for path in resampled_paths)
    )


def get_last_processed_file(output_dir: str) -> Optional[str]:
    """
    Get the path of the last processed file by checking modification times in the final step directory.

    Args:
        output_dir: Base output directory

    Returns:
        Path to the last processed file or None if no files were processed
    """
    # The last step is resampling (step 3)
    step3_dir = create_step_directory(output_dir, 3, "resampling")
    if not os.path.exists(step3_dir):
        return None

    # Get all resampled files and sort by modification time
    resampled_files = []
    for root, _, files in os.walk(step3_dir):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                file_path = os.path.join(root, file)
                resampled_files.append((file_path, os.path.getmtime(file_path)))

    if not resampled_files:
        return None

    # Sort by modification time and get the most recent
    resampled_files.sort(key=lambda x: x[1])
    return resampled_files[-1][0]


def process_directory(
    input_dir: str,
    output_dir: str,
    template_file: str,
    show_progress: bool = True,
    include_dirs_regex: Optional[List[str]] = None
) -> None:
    """
    Process all NIFTI files in the input directory.

    Args:
        input_dir: Directory containing input MRI images
        output_dir: Base directory to save output files
        template_file: Path to template image for registration
        show_progress: Whether to show progress bars
        include_dirs_regex: Optional list of regex patterns to filter first-level directories

    Raises:
        PreprocessingError: If processing fails
    """
    # Find all NIFTI files in the input directory and subdirectories
    nifti_files = find_nifti_files(input_dir, include_dirs_regex)

    if not nifti_files:
        raise PreprocessingError(f"No NIFTI files (.nii or .nii.gz) found in {input_dir}")

    logging.info(f"Found {len(nifti_files)} NIFTI files to process.")

    # Get the last processed file (if any)
    last_processed = get_last_processed_file(output_dir)
    if last_processed:
        logging.info(f"Found last processed file: {last_processed}")
        # Remove the last processed file from the list to reprocess it
        resampling_dir = create_step_directory(output_dir, 3, "resampling")
        last_processed_input = os.path.join(
            input_dir, os.path.relpath(last_processed, resampling_dir)
        )
        if last_processed_input in nifti_files:
            nifti_files.remove(last_processed_input)
            logging.info("Will reprocess the last file to ensure it's not corrupted")

    # Filter out already processed files
    files_to_process = []
    for nifti_file in nifti_files:
        if not is_file_fully_processed(nifti_file, output_dir, input_dir):
            files_to_process.append(nifti_file)
        else:
            logging.info(f"Skipping already processed file: {nifti_file}")

    if not files_to_process:
        logging.info("No new files to process. All files have been processed.")
        return

    logging.info(f"Processing {len(files_to_process)} new files.")
    input("Press Enter to continue...")

    # Process each file with progress bar if enabled
    file_iterator = (
        tqdm(files_to_process, desc="Processing MRI files", unit="file")
        if show_progress
        else files_to_process
    )

    for nifti_file in file_iterator:
        rel_path = get_relative_path(nifti_file, input_dir)
        logging.info(f"Processing file: {rel_path}")
        preprocess_mri_file(nifti_file, input_dir, output_dir, template_file, show_progress)

    # If there was a last processed file, reprocess it
    if last_processed:
        resampling_dir = create_step_directory(output_dir, 3, "resampling")
        last_processed_input = os.path.join(
            input_dir, os.path.relpath(last_processed, resampling_dir)
        )
        if os.path.exists(last_processed_input):
            logging.info("Reprocessing last file to ensure it's not corrupted...")
            preprocess_mri_file(last_processed_input, input_dir, output_dir, template_file, show_progress)

    # Clean up temporary files
    temp_dir = os.path.join(os.path.dirname(output_dir), "temp")
    if os.path.exists(temp_dir):
        logging.info("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_dir)

    logging.info("Preprocessing completed.")


def main() -> None:
    """Main entry point of the script."""
    try:
        # Set up logging
        setup_logging()

        # Parse command-line arguments
        args = parse_arguments()

        # Check if dependencies are installed
        check_dependencies()

        # Prepare include_dirs_regex list
        include_dirs_regex_list = None
        if args.include_dirs_regex:
            include_dirs_regex_list = [pattern.strip() for pattern in args.include_dirs_regex.split(',')]

        # Process all files in the input directory
        process_directory(
            input_dir=args.input,
            output_dir=args.output,
            template_file=args.template,
            show_progress=not args.no_progress,
            include_dirs_regex=include_dirs_regex_list
        )

        sys.exit(0)
    except PreprocessingError as e:
        logging.error(f"Preprocessing error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
