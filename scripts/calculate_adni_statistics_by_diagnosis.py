#!/usr/bin/env python3
"""
ADNI Dataset Statistics Calculator - Grouped by Diagnosis

This script calculates statistics grouped by diagnosis categories (CN, AD, MCI, etc.).
For each diagnosis group, it provides gender, age, and MMSE statistics.
The CSV file should contain all necessary columns including MMSE.

Usage:
    python scripts/calculate_adni_statistics_by_diagnosis.py <csv_file_path>

Example:
    python scripts/calculate_adni_statistics_by_diagnosis.py "data/ADNI/3T_bl_org_MRI_UniqueSID_1220images_fullmetadata.csv"
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')


def check_mmse_availability(df: pd.DataFrame) -> bool:
    """Check if MMSE column is available in the dataset."""
    mmse_cols = ['MMSE', 'MMSE_bl']
    available_cols = [col for col in mmse_cols if col in df.columns and df[col].notna().sum() > 0]

    if available_cols:
        print(f"✓ MMSE data available in column(s): {', '.join(available_cols)}")
        return True
    else:
        print("✗ No MMSE data found in the dataset")
        return False


def load_main_csv(csv_path: str) -> pd.DataFrame:
    """Load the main ADNI CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded main CSV: {len(df)} records")
        return df
    except FileNotFoundError:
        print(f"✗ Error: CSV file not found at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading CSV file: {e}")
        sys.exit(1)


def get_best_mmse_column(df: pd.DataFrame) -> str:
    """Determine the best MMSE column to use."""
    mmse_cols = ['MMSE', 'MMSE_bl']

    for col in mmse_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            mmse_count = df[col].notna().sum()
            print(f"✓ Using MMSE column '{col}' with {mmse_count} available values")
            return col

    return None


def calculate_diagnosis_group_statistics(df: pd.DataFrame, diagnosis_col: str, mmse_col: str = None) -> Dict[str, Dict[str, Any]]:
    """Calculate statistics for each diagnosis group."""
    diagnosis_stats = {}

    # Get unique diagnosis categories
    diagnoses = df[diagnosis_col].dropna().unique()
    print(f"✓ Found diagnosis categories: {', '.join(sorted(diagnoses))}")

    for dx in sorted(diagnoses):
        if pd.isna(dx):
            continue

        dx_group = df[df[diagnosis_col] == dx].copy()
        stats = {}

        # Basic counts
        stats['total_cases'] = len(dx_group)
        stats['unique_subjects'] = dx_group['subject_id'].nunique() if 'subject_id' in dx_group.columns else len(dx_group)

        # Gender statistics
        if 'PTGENDER' in dx_group.columns:
            gender_counts = dx_group['PTGENDER'].value_counts()
            stats['male_count'] = gender_counts.get('Male', 0)
            stats['female_count'] = gender_counts.get('Female', 0)
            stats['gender_total'] = stats['male_count'] + stats['female_count']
        else:
            stats['male_count'] = 0
            stats['female_count'] = 0
            stats['gender_total'] = 0

        # Age statistics
        if 'AGE' in dx_group.columns:
            age_data = dx_group['AGE'].dropna()
            if len(age_data) > 0:
                stats['age_count'] = len(age_data)
                stats['age_mean'] = age_data.mean()
                stats['age_std'] = age_data.std()
                stats['age_min'] = age_data.min()
                stats['age_max'] = age_data.max()
            else:
                stats['age_count'] = 0
        else:
            stats['age_count'] = 0

        # MMSE statistics
        if mmse_col and mmse_col in dx_group.columns:
            mmse_data = dx_group[mmse_col].dropna()
            if len(mmse_data) > 0:
                stats['mmse_count'] = len(mmse_data)
                stats['mmse_mean'] = mmse_data.mean()
                stats['mmse_std'] = mmse_data.std()
                stats['mmse_min'] = mmse_data.min()
                stats['mmse_max'] = mmse_data.max()
            else:
                stats['mmse_count'] = 0
        else:
            stats['mmse_count'] = 0

        diagnosis_stats[dx] = stats

    return diagnosis_stats


def print_diagnosis_statistics_table(stats: Dict[str, Dict[str, Any]], dataset_name: str = "ADNI"):
    """Print diagnosis statistics in a formatted table."""

    print("\n" + "="*120)
    print("ADNI DATASET STATISTICS BY DIAGNOSIS")
    print("="*120)

    # Header
    print(f"\n{'Dataset':<10} {'Category':<10} {'Subjects':<10} {'Gender':<15} {'Age(years)':<20} {'MMSE':<15}")
    print("-" * 120)

    # Data rows for each diagnosis
    for dx, dx_stats in sorted(stats.items()):
        # Number of subjects
        subjects_str = f"{dx_stats['unique_subjects']}"

        # Format gender as "male_count/female_count"
        if dx_stats['gender_total'] > 0:
            gender_str = f"{dx_stats['male_count']}/{dx_stats['female_count']}"
        else:
            gender_str = "N/A"

        # Format age as "mean ± std"
        if dx_stats['age_count'] > 0:
            age_str = f"{dx_stats['age_mean']:.1f} ± {dx_stats['age_std']:.1f}"
        else:
            age_str = "N/A"

        # Format MMSE as "mean ± std"
        if dx_stats['mmse_count'] > 0:
            mmse_str = f"{dx_stats['mmse_mean']:.1f} ± {dx_stats['mmse_std']:.1f}"
        else:
            mmse_str = "N/A"

        print(f"{dataset_name:<10} {dx:<10} {subjects_str:<10} {gender_str:<15} {age_str:<20} {mmse_str:<15}")


def print_copy_paste_table(stats: Dict[str, Dict[str, Any]], dataset_name: str = "ADNI"):
    """Print table format ready for copying to Word/Excel."""

    print("\n" + "="*80)
    print("COPY-PASTE READY TABLE")
    print("="*80)

    print("Dataset\tCategory\tSubjects\tGender\tAge(years)\tMMSE")

    for dx, dx_stats in sorted(stats.items()):
        # Number of subjects
        subjects_str = f"{dx_stats['unique_subjects']}"

        # Format gender as "male_count/female_count"
        if dx_stats['gender_total'] > 0:
            gender_str = f"{dx_stats['male_count']}/{dx_stats['female_count']}"
        else:
            gender_str = "N/A"

        # Format age as "mean ± std"
        if dx_stats['age_count'] > 0:
            age_str = f"{dx_stats['age_mean']:.1f} ± {dx_stats['age_std']:.1f}"
        else:
            age_str = "N/A"

        # Format MMSE as "mean ± std"
        if dx_stats['mmse_count'] > 0:
            mmse_str = f"{dx_stats['mmse_mean']:.1f} ± {dx_stats['mmse_std']:.1f}"
        else:
            mmse_str = "N/A"

        print(f"{dataset_name}\t{dx}\t{subjects_str}\t{gender_str}\t{age_str}\t{mmse_str}")


def print_detailed_statistics(stats: Dict[str, Dict[str, Any]]):
    """Print detailed statistics for each diagnosis group."""

    print("\n" + "="*80)
    print("DETAILED STATISTICS BY DIAGNOSIS")
    print("="*80)

    for dx, dx_stats in sorted(stats.items()):
        print(f"\n📊 {dx} (n={dx_stats['total_cases']:,})")
        print("-" * 40)

        print(f"Unique Subjects: {dx_stats['unique_subjects']:,}")

        if dx_stats['gender_total'] > 0:
            male_pct = (dx_stats['male_count'] / dx_stats['gender_total']) * 100
            female_pct = (dx_stats['female_count'] / dx_stats['gender_total']) * 100
            print(f"Gender: {dx_stats['male_count']} Male ({male_pct:.1f}%), {dx_stats['female_count']} Female ({female_pct:.1f}%)")
        else:
            print("Gender: No data available")

        if dx_stats['age_count'] > 0:
            print(f"Age: {dx_stats['age_mean']:.2f} ± {dx_stats['age_std']:.2f} years (range: {dx_stats['age_min']:.1f}-{dx_stats['age_max']:.1f})")
        else:
            print("Age: No data available")

        if dx_stats['mmse_count'] > 0:
            print(f"MMSE: {dx_stats['mmse_mean']:.2f} ± {dx_stats['mmse_std']:.2f} (range: {dx_stats['mmse_min']:.1f}-{dx_stats['mmse_max']:.1f})")
        else:
            print("MMSE: No data available")


def main():
    """Main function to run the diagnosis-grouped statistics calculation."""
    parser = argparse.ArgumentParser(description='Calculate ADNI dataset statistics grouped by diagnosis')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('--dataset-name', default='ADNI',
                        help='Dataset name for table output (default: ADNI)')

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"✗ Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    print("ADNI Dataset Statistics Calculator - Grouped by Diagnosis")
    print("=" * 60)

    # Load data
    print("\n📁 Loading data...")
    main_df = load_main_csv(args.csv_file)

    # Check for MMSE data availability
    print("\n🔍 Checking MMSE data availability...")
    mmse_available = check_mmse_availability(main_df)
    mmse_col = get_best_mmse_column(main_df) if mmse_available else None

    # Determine which diagnosis column to use
    diagnosis_cols = ['DX', 'DX_bl', 'DIAGNOSIS']
    dx_col = None

    for col in diagnosis_cols:
        if col in main_df.columns and main_df[col].notna().sum() > 0:
            dx_col = col
            print(f"✓ Using diagnosis column: {dx_col}")
            break

    if not dx_col:
        print("✗ Error: No valid diagnosis column found in the dataset")
        print("Expected columns: DX, DX_bl, or DIAGNOSIS")
        sys.exit(1)

    # Calculate statistics grouped by diagnosis
    print("\n📊 Calculating statistics by diagnosis...")
    diagnosis_stats = calculate_diagnosis_group_statistics(main_df, dx_col, mmse_col)

    # Print results
    print_diagnosis_statistics_table(diagnosis_stats, args.dataset_name)
    print_copy_paste_table(diagnosis_stats, args.dataset_name)
    print_detailed_statistics(diagnosis_stats)

    print("\n" + "="*80)
    print("✓ Diagnosis-grouped statistics calculation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
