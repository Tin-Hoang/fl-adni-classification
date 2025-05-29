#!/bin/bash

# Recursively rename ADNI files to standardized format.
# Usage: ./rename_adni_files.sh <root_dir> [--dry-run]
# Example: ./rename_adni_files.sh data/ADNI/3T_bl_org_MRI_2_NIfTI/ --dry-run

set -e

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <root_dir> [--dry-run)"
    exit 1
fi

ROOT_DIR="$1"
DRY_RUN=false
if [ "$2" == "--dry-run" ]; then
    DRY_RUN=true
fi

# Find all .nii.gz and .json files recursively
# Limit to 5 results for dry run in dry run mode
FIND_CMD="find \"$ROOT_DIR\" -type f \( -iname \"*.nii.gz\" -o -iname \"*.json\" \)"

if $DRY_RUN; then
    eval "$FIND_CMD" | head -n 5 | while read -r FILE;
    do
        # Extract the path components relative to the first 'ADNI' directory
        # Expected structure after ADNI: <subject_id>/<scan_type>/<session>/<image_id>/<file>
        ADNI_REL_PATH=$(echo "$FILE" | sed -e 's/.*\/ADNI\///')

        # Check if the path contains 'ADNI'
        if [ "$FILE" == "$ADNI_REL_PATH" ]; then
             echo "Skipping (ADNI directory not found in path): $FILE"
             continue
        fi

        # Split the relative path into components
        IFS='/' read -r SUBJECT_ID SCAN_TYPE SESSION IMAGE_ID FILENAME <<< "$ADNI_REL_PATH"

        # If any crucial component is missing, skip
        if [ -z "$SUBJECT_ID" ] || [ -z "$SCAN_TYPE" ] || [ -z "$IMAGE_ID" ] || [ -z "$FILENAME" ]; then
            echo "Skipping (unexpected structure after ADNI): $FILE"
            continue
        fi

        # Extract timestamp (14 digits) from filename
        TIMESTAMP=$(echo "$FILENAME" | grep -oE '[0-9]{14}')
        if [ -z "$TIMESTAMP" ]; then
            echo "Skipping (no timestamp found in filename): $FILE"
            continue
        fi

        # Get file extension (handle .nii.gz and .json)
        EXT="${FILENAME##*.}"
        if [[ "$FILENAME" == *.nii.gz ]]; then
            EXT="nii.gz"
        fi

        # Compose new filename with fixed ADNI prefix and correct components
        NEW_FILENAME="ADNI_${SUBJECT_ID}_${SCAN_TYPE}_${TIMESTAMP}_${IMAGE_ID}.${EXT}"
        NEW_PATH="$(dirname "$FILE")/$NEW_FILENAME"

        if [ "$FILE" == "$NEW_PATH" ]; then
            # Already named correctly
            continue
        fi

        echo "[DRY RUN] Would rename: $FILE -> $NEW_PATH"

    done
else
    eval "$FIND_CMD" | while read -r FILE;
    do
        # Extract the path components relative to the first 'ADNI' directory
        # Expected structure after ADNI: <subject_id>/<scan_type>/<session>/<image_id>/<file>
        ADNI_REL_PATH=$(echo "$FILE" | sed -e 's/.*\/ADNI\///')

        # Check if the path contains 'ADNI'
        if [ "$FILE" == "$ADNI_REL_PATH" ]; then
             echo "Skipping (ADNI directory not found in path): $FILE"
             continue
        fi

        # Split the relative path into components
        IFS='/' read -r SUBJECT_ID SCAN_TYPE SESSION IMAGE_ID FILENAME <<< "$ADNI_REL_PATH"

        # If any crucial component is missing, skip
        if [ -z "$SUBJECT_ID" ] || [ -z "$SCAN_TYPE" ] || [ -z "$IMAGE_ID" ] || [ -z "$FILENAME" ]; then
            echo "Skipping (unexpected structure after ADNI): $FILE"
            continue
        fi

        # Extract timestamp (14 digits) from filename
        TIMESTAMP=$(echo "$FILENAME" | grep -oE '[0-9]{14}')
        if [ -z "$TIMESTAMP" ]; then
            echo "Skipping (no timestamp found in filename): $FILE"
            continue
        fi

        # Get file extension (handle .nii.gz and .json)
        EXT="${FILENAME##*.}"
        if [[ "$FILENAME" == *.nii.gz ]]; then
            EXT="nii.gz"
        fi

        # Compose new filename with fixed ADNI prefix and correct components
        NEW_FILENAME="ADNI_${SUBJECT_ID}_${SCAN_TYPE}_${TIMESTAMP}_${IMAGE_ID}.${EXT}"
        NEW_PATH="$(dirname "$FILE")/$NEW_FILENAME"

        if [ "$FILE" == "$NEW_PATH" ]; then
            # Already named correctly
            continue
        fi

        echo "Renaming: $FILE -> $NEW_PATH"
        # Uncomment the line below to actually perform the rename
        mv "$FILE" "$NEW_PATH"
    done
fi

echo "Done."
