#!/usr/bin/env python3
"""
Script to convert 3D NIfTI images to 2D slices for debugging and corruption checking.

This script recursively finds all .nii and .nii.gz files in a directory,
extracts middle slices in three directions (axial, coronal, sagittal),
and saves them as 2D PNG images for easy inspection.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import nibabel as nib
from tqdm import tqdm


def find_nifti_files(root_dir: str) -> List[Path]:
    """Find all NIfTI files recursively in a directory.

    Args:
        root_dir: Root directory to search

    Returns:
        List of paths to NIfTI files
    """
    root_path = Path(root_dir)
    nifti_files = []

    # Find .nii and .nii.gz files
    for pattern in ['**/*.nii', '**/*.nii.gz']:
        nifti_files.extend(root_path.glob(pattern))

    return sorted(nifti_files)


def load_nifti_safely(file_path: Path) -> Optional[np.ndarray]:
    """Safely load a NIfTI file and return the data array.

    Args:
        file_path: Path to the NIfTI file

    Returns:
        Numpy array of the image data, or None if loading failed
    """
    try:
        img = nib.load(str(file_path))
        data = img.get_fdata()

        # Check if data is valid
        if data.size == 0:
            print(f"WARNING: {file_path.name} contains empty data")
            return None

        if np.all(np.isnan(data)):
            print(f"WARNING: {file_path.name} contains all NaN values")
            return None

        return data

    except Exception as e:
        print(f"ERROR: Failed to load {file_path.name}: {str(e)}")
        return None


def normalize_image_data(data: np.ndarray) -> np.ndarray:
    """Normalize image data to 0-255 range for visualization.

    Args:
        data: Raw image data

    Returns:
        Normalized image data
    """
    # Handle NaN values
    data = np.nan_to_num(data, nan=0.0)

    # Normalize to 0-1 range
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max == data_min:
        # Handle constant images
        return np.zeros_like(data)

    normalized = (data - data_min) / (data_max - data_min)

    # Convert to 0-255 range
    return (normalized * 255).astype(np.uint8)


def extract_middle_slices(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract middle slices in three directions from 3D data.

    Args:
        data: 3D numpy array

    Returns:
        Tuple of (axial, coronal, sagittal) slices
    """
    # Get middle indices for each dimension
    mid_z = data.shape[0] // 2  # Axial (top-down view)
    mid_y = data.shape[1] // 2  # Coronal (front-back view)
    mid_x = data.shape[2] // 2  # Sagittal (left-right view)

    # Extract slices
    axial_slice = data[mid_z, :, :]      # Shape: (height, width)
    coronal_slice = data[:, mid_y, :]    # Shape: (depth, width)
    sagittal_slice = data[:, :, mid_x]   # Shape: (depth, height)

    return axial_slice, coronal_slice, sagittal_slice


def save_slice_as_image(slice_data: np.ndarray, output_path: Path, title: str = "") -> bool:
    """Save a 2D slice as a PNG image.

    Args:
        slice_data: 2D numpy array
        output_path: Path to save the image
        title: Title for the image

    Returns:
        True if successful, False otherwise
    """
    try:
        # Normalize the data
        normalized_data = normalize_image_data(slice_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(normalized_data, cmap='gray', aspect='equal')
        ax.set_title(title)
        ax.axis('off')

        # Save with high DPI for better quality
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return True

    except Exception as e:
        print(f"ERROR: Failed to save image {output_path}: {str(e)}")
        plt.close()
        return False


def process_nifti_file(file_path: Path, output_dir: Path) -> dict:
    """Process a single NIfTI file and extract 2D slices.

    Args:
        file_path: Path to the NIfTI file
        output_dir: Directory to save output images

    Returns:
        Dictionary with processing results
    """
    result = {
        'file': file_path.name,
        'success': False,
        'error': None,
        'slices_saved': 0,
        'shape': None
    }

    # Load the NIfTI file
    data = load_nifti_safely(file_path)
    if data is None:
        result['error'] = 'Failed to load file'
        return result

    # Check if it's 3D
    if len(data.shape) < 3:
        result['error'] = f'Not a 3D image (shape: {data.shape})'
        return result

    # If 4D, take the first volume
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
        print(f"INFO: {file_path.name} is 4D, using first volume")

    result['shape'] = data.shape

    # Extract middle slices
    try:
        axial_slice, coronal_slice, sagittal_slice = extract_middle_slices(data)
    except Exception as e:
        result['error'] = f'Failed to extract slices: {str(e)}'
        return result

    # Generate output filenames
    base_name = file_path.stem
    if base_name.endswith('.nii'):  # Remove .nii from .nii.gz files
        base_name = base_name[:-4]

    slice_info = [
        (axial_slice, f"{base_name}_axial.png", f"Axial slice (Z={data.shape[0]//2})"),
        (coronal_slice, f"{base_name}_coronal.png", f"Coronal slice (Y={data.shape[1]//2})"),
        (sagittal_slice, f"{base_name}_sagittal.png", f"Sagittal slice (X={data.shape[2]//2})")
    ]

    # Save each slice
    for slice_data, filename, title in slice_info:
        output_path = output_dir / filename
        if save_slice_as_image(slice_data, output_path, title):
            result['slices_saved'] += 1

    result['success'] = result['slices_saved'] > 0
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D NIfTI images to 2D slices for debugging"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing NIfTI files"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for 2D slice images"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory {input_dir} does not exist")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NIfTI files
    print(f"Searching for NIfTI files in {input_dir}...")
    nifti_files = find_nifti_files(str(input_dir))

    if not nifti_files:
        print("No NIfTI files found!")
        return 1

    print(f"Found {len(nifti_files)} NIfTI files")

    # Limit files for testing if specified
    if args.max_files:
        nifti_files = nifti_files[:args.max_files]
        print(f"Processing first {len(nifti_files)} files")

    # Process each file
    results = []
    successful_files = 0
    total_slices = 0

    print(f"\nProcessing files...")
    for file_path in tqdm(nifti_files, desc="Converting"):
        result = process_nifti_file(file_path, output_dir)
        results.append(result)

        if result['success']:
            successful_files += 1
            total_slices += result['slices_saved']
        else:
            print(f"FAILED: {result['file']} - {result['error']}")

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total files processed: {len(nifti_files)}")
    print(f"Successful files: {successful_files}")
    print(f"Failed files: {len(nifti_files) - successful_files}")
    print(f"Total slices saved: {total_slices}")
    print(f"Output directory: {output_dir}")

    # Print failed files details
    failed_files = [r for r in results if not r['success']]
    if failed_files:
        print(f"\n=== FAILED FILES ===")
        for result in failed_files:
            print(f"- {result['file']}: {result['error']}")

    # Print some successful files info
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n=== SAMPLE SUCCESSFUL FILES ===")
        for result in successful_results[:5]:  # Show first 5
            print(f"- {result['file']}: shape {result['shape']}, {result['slices_saved']} slices saved")
        if len(successful_results) > 5:
            print(f"... and {len(successful_results) - 5} more")

    return 0


if __name__ == "__main__":
    exit(main())
