#!/bin/bash

# Usage: ./convert_dicom_to_nifti.sh <input_root_dir> <output_root_dir> [debug]
# Example: ./convert_dicom_to_nifti.sh data/ADNI data/ADNI_NIfTI
# Debug: ./convert_dicom_to_nifti.sh data/ADNI data/ADNI_NIfTI debug

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_root_dir> <output_root_dir> [debug]"
    exit 1
fi

INPUT_ROOT="$1"
OUTPUT_ROOT="$2"
DEBUG_MODE=0
if [ "$3" = "debug" ]; then
    DEBUG_MODE=1
    echo "Debug mode enabled: Logging DICOM tags for each directory"
fi

LOG_FILE="conversion_errors.log"
DEBUG_LOG="dicom_tags.log"

# Create log files
echo "Conversion errors (if any) will be logged to $LOG_FILE"
: > "$LOG_FILE"
if [ $DEBUG_MODE -eq 1 ]; then
    echo "DICOM tag debug info will be logged to $DEBUG_LOG"
    : > "$DEBUG_LOG"
fi

# Function to infer series description from folder path if DICOM extraction fails
infer_series_from_path() {
    local path="$1"
    IFS='/' read -ra PATH_PARTS <<< "$path"
    # Look for a folder name that might indicate the series (e.g., MP-RAGE_REPEAT)
    for part in "${PATH_PARTS[@]}"; do
        if [[ "$part" =~ ^(MP-RAGE|MPRAGE|FLAIR|T2|IR-FSPGR|Accelerated|B1-Calibration) ]]; then
            echo "$part"
            return
        fi
    done
    echo "Unknown"
}

# Count total DICOM images in input root (before conversion)
TOTAL_DICOM_COUNT=$(find "$INPUT_ROOT" -type f \( -iname "*.dcm" -o -iname "*.ima" \) | wc -l)
echo "Total DICOM images in input: $TOTAL_DICOM_COUNT"

# Find all directories containing DICOM files
DICOM_DIRS=( $(find "$INPUT_ROOT" -type d) )
TOTAL_DIRS=${#DICOM_DIRS[@]}
CURRENT_DIR=0
SKIPPED_DIRS=0
FAILED_DIRS=0

for DICOM_DIR in "${DICOM_DIRS[@]}"; do
    CURRENT_DIR=$((CURRENT_DIR + 1))
    # Check if directory contains DICOM files (by extension or by file magic)
    if find "$DICOM_DIR" -maxdepth 1 -type f \( -iname "*.dcm" -o -iname "*.ima" -o -empty \) | grep -q .; then
        # Compute relative path starting from the subject ID level
        # Expected structure: data/ADNI/<study_info>/ADNI/<subject_id>/<scan_info>/<date_time>/<image_id>
        FULL_REL_PATH="${DICOM_DIR#$INPUT_ROOT/}"
        IFS='/' read -ra PATH_PARTS <<< "$FULL_REL_PATH"
        # Find the index of the second "ADNI" to start from subject ID
        ADNI_COUNT=0
        SUBJECT_INDEX=-1
        for i in "${!PATH_PARTS[@]}"; do
            if [ "${PATH_PARTS[$i]}" = "ADNI" ]; then
                ADNI_COUNT=$((ADNI_COUNT + 1))
                if [ $ADNI_COUNT -eq 2 ]; then
                    SUBJECT_INDEX=$((i + 1))
                    break
                fi
            fi
        done

        if [ $SUBJECT_INDEX -eq -1 ] || [ $SUBJECT_INDEX -ge ${#PATH_PARTS[@]} ]; then
            echo "Error: Could not locate subject ID in path: $DICOM_DIR" >> "$LOG_FILE"
            FAILED_DIRS=$((FAILED_DIRS + 1))
            continue
        fi

        # Reconstruct REL_PATH starting from subject ID
        REL_PATH=$(IFS='/'; echo "${PATH_PARTS[*]:$SUBJECT_INDEX}")

        # Compute output directory
        OUT_DIR="$OUTPUT_ROOT/$REL_PATH"

        # Check if output directory already contains NIfTI files
        if [ -d "$OUT_DIR" ] && find "$OUT_DIR" -type f \( -iname "*.nii" -o -iname "*.nii.gz" \) 2>/dev/null | grep -q .; then
            echo "Skipping $CURRENT_DIR/$TOTAL_DIRS: $DICOM_DIR (NIfTI files already exist in $OUT_DIR)"
            SKIPPED_DIRS=$((SKIPPED_DIRS + 1))
            continue
        fi

        echo "Processing $CURRENT_DIR/$TOTAL_DIRS: $DICOM_DIR"
        mkdir -p "$OUT_DIR"
        echo "Converting: $DICOM_DIR -> $OUT_DIR"

        # # Extract image ID from last subfolder
        IMAGE_ID="${PATH_PARTS[-1]}" # e.g., I41838
        if [[ ! "$IMAGE_ID" =~ ^I[0-9]+$ ]]; then
            echo "Error: Invalid image ID format in path: $IMAGE_ID" >> "$LOG_FILE"
            FAILED_DIRS=$((FAILED_DIRS + 1))
            continue
        fi

        # Run dcm2niix with ADNI naming convention
        # Use %m for modality, %t for timestamp, %s for series number
        if dcm2niix -z y -b y -f "ADNI_%i_%d_%t_${IMAGE_ID}" -o "$OUT_DIR" "$DICOM_DIR" 2>> "$LOG_FILE"; then
            echo "Success: Converted $DICOM_DIR"
        else
            echo "Error: Failed to convert $DICOM_DIR (see $LOG_FILE for details)"
            FAILED_DIRS=$((FAILED_DIRS + 1))
            echo "Failed directory: $DICOM_DIR" >> "$LOG_FILE"
            # Remove incomplete output directory to avoid confusion
            rm -rf "$OUT_DIR"
        fi
    fi
done

# Count total NIfTI images in output root (after conversion)
TOTAL_NIFTI_COUNT=$(find "$OUTPUT_ROOT" -type f \( -iname "*.nii" -o -iname "*.nii.gz" \) | wc -l)
echo "Total NIfTI images in output: $TOTAL_NIFTI_COUNT"
echo "Skipped directories (existing NIfTI): $SKIPPED_DIRS"
echo "Failed directories: $FAILED_DIRS"

if [ "$TOTAL_DICOM_COUNT" -ne "$TOTAL_NIFTI_COUNT" ]; then
    echo "WARNING: Total number of DICOM and NIfTI images do not match!"
    echo "Check $LOG_FILE for failed conversions."
fi

if [ "$FAILED_DIRS" -gt 0 ]; then
    echo "Some conversions failed. Review $LOG_FILE for details."
fi

if [ $DEBUG_MODE -eq 1 ]; then
    echo "Debug DICOM tags logged to $DEBUG_LOG"
fi
