#!/bin/bash

# Usage: ./convert_dicom_to_nifti.sh <input_root_dir> <output_root_dir>
# Example: ./convert_dicom_to_nifti.sh data/ADNI data/ADNI_NIfTI

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_root_dir> <output_root_dir>"
    exit 1
fi

INPUT_ROOT="$1"
OUTPUT_ROOT="$2"

# Count total DICOM images in input root (before conversion)
TOTAL_DICOM_COUNT=$(find "$INPUT_ROOT" -type f \( -iname "*.dcm" -o -iname "*.ima" \) | wc -l)
echo "Total DICOM images in input: $TOTAL_DICOM_COUNT"

# Find all directories containing DICOM files (assuming .dcm or no extension)
DICOM_DIRS=( $(find "$INPUT_ROOT" -type d) )
TOTAL_DIRS=${#DICOM_DIRS[@]}
CURRENT_DIR=0
for DICOM_DIR in "${DICOM_DIRS[@]}"; do
    CURRENT_DIR=$((CURRENT_DIR + 1))
    # Check if directory contains DICOM files (by extension or by file magic)
    if find "$DICOM_DIR" -maxdepth 1 -type f \( -iname "*.dcm" -o -iname "*.ima" -o -empty \) | grep -q .; then
        echo "Processing $CURRENT_DIR/$TOTAL_DIRS: $DICOM_DIR"
        # Compute relative path
        REL_PATH="${DICOM_DIR#$INPUT_ROOT/}"
        OUT_DIR="$OUTPUT_ROOT/$REL_PATH"
        mkdir -p "$OUT_DIR"
        echo "Converting: $DICOM_DIR -> $OUT_DIR"
        dcm2niix -z y -o "$OUT_DIR" "$DICOM_DIR"
    fi
done

# Count total NIfTI images in output root (after conversion)
TOTAL_NIFTI_COUNT=$(find "$OUTPUT_ROOT" -type f \( -iname "*.nii" -o -iname "*.nii.gz" \) | wc -l)
echo "Total NIfTI images in output: $TOTAL_NIFTI_COUNT"

if [ "$TOTAL_DICOM_COUNT" -ne "$TOTAL_NIFTI_COUNT" ]; then
    echo "WARNING: Total number of DICOM and NIfTI images do not match!"
fi
