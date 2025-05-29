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

# Function to map series descriptions to ADNI abbreviations
map_series_description() {
    local desc="$1"
    case "$desc" in
        "Accelerated Sagittal MPRAGE"|"Sagittal MPRAGE"|"MPRAGE"|"MP-RAGE")
            echo "MPRAGE"
            ;;
        "Accelerated Sag IR-FSPGR"|"Sag IR-FSPGR"|"IR-FSPGR")
            echo "IR-FSPGR"
            ;;
        "FLAIR")
            echo "FLAIR"
            ;;
        "T2"|"T2-weighted")
            echo "T2"
            ;;
        *)
            # Use the description as-is if no mapping exists
            echo "$desc" | sed 's/[^a-zA-Z0-9_-]/_/g' # Replace spaces/special chars with underscores
            ;;
    esac
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
        # Compute relative path and output directory
        REL_PATH="${DICOM_DIR#$INPUT_ROOT/}"
        OUT_DIR="$OUTPUT_ROOT/$REL_PATH"

        # Check if output directory already contains NIfTI files
        if find "$OUT_DIR" -type f \( -iname "*.nii" -o -iname "*.nii.gz" \) | grep -q .; then
            echo "Skipping $CURRENT_DIR/$TOTAL_DIRS: $DICOM_DIR (NIfTI files already exist in $OUT_DIR)"
            SKIPPED_DIRS=$((SKIPPED_DIRS + 1))
            continue
        fi

        echo "Processing $CURRENT_DIR/$TOTAL_DIRS: $DICOM_DIR"
        mkdir -p "$OUT_DIR"
        echo "Converting: $DICOM_DIR -> $OUT_DIR"

        # Extract subject ID from folder path (second subfolder after ADNI)
        IFS='/' read -ra PATH_PARTS <<< "$REL_PATH"
        if [ ${#PATH_PARTS[@]} -lt 2 ]; then
            echo "Error: Could not extract subject ID from path: $DICOM_DIR" >> "$LOG_FILE"
            FAILED_DIRS=$((FAILED_DIRS + 1))
            continue
        fi
        SUBJECT_ID="${PATH_PARTS[1]}" # e.g., 052_S_1250

        # Extract image ID from last subfolder
        IMAGE_ID="${PATH_PARTS[-1]}" # e.g., I41844
        if [[ ! "$IMAGE_ID" =~ ^I[0-9]+$ ]]; then
            echo "Error: Invalid image ID format in path: $IMAGE_ID" >> "$LOG_FILE"
            FAILED_DIRS=$((FAILED_DIRS + 1))
            continue
        fi

        # Debug mode: Log DICOM tags
        if [ $DEBUG_MODE -eq 1 ]; then
            echo "Debugging DICOM tags for: $DICOM_DIR" >> "$DEBUG_LOG"
            dcm2niix -v y -o "/tmp" -f "debug_%p_%c_%i_%u_%n_%s_%d_%m" "$DICOM_DIR" >> "$DEBUG_LOG" 2>&1
        fi

        # Get DICOM Patient Name to verify subject ID
        PATIENT_NAME=$(dcm2niix -v y -o "/tmp" "$DICOM_DIR" | grep "Patient Name" | awk -F': ' '{print $2}' | tr -d ' ')
        if [ -z "$PATIENT_NAME" ]; then
            echo "Warning: Could not extract Patient Name from DICOM: $DICOM_DIR" >> "$LOG_FILE"
        elif [ "$PATIENT_NAME" != "$SUBJECT_ID" ]; then
            echo "Warning: Subject ID mismatch in $DICOM_DIR: folder ($SUBJECT_ID) vs DICOM ($PATIENT_NAME)" >> "$LOG_FILE"
        fi

        # Get series description and map it
        SERIES_DESC=$(dcm2niix -v y -o "/tmp" "$DICOM_DIR" | grep "Series Description" | awk -F': ' '{print $2}' | tr -d ' ')
        if [ -z "$SERIES_DESC" ]; then
            SERIES_DESC="Unknown"
        fi
        MAPPED_SERIES_DESC=$(map_series_description "$SERIES_DESC")

        # Get modality for logging (optional, for verification)
        MODALITY=$(dcm2niix -v y -o "/tmp" "$DICOM_DIR" | grep "Modality" | awk -F': ' '{print $2}' | tr -d ' ')
        if [ -z "$MODALITY" ]; then
            echo "Warning: Could not extract Modality from DICOM: $DICOM_DIR" >> "$LOG_FILE"
            MODALITY="Unknown"
        fi

        # Run dcm2niix with ADNI naming convention
        # Use %m for modality, %t for timestamp, %s for series number
        if dcm2niix -z y -b y -f "ADNI_${SUBJECT_ID}_%m_${MAPPED_SERIES_DESC}_Br_%t_S%s_${IMAGE_ID}" -o "$OUT_DIR" "$DICOM_DIR" 2>> "$LOG_FILE"; then
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
