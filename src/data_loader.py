"""
Data-loading and inspection module.

Handles loading the RT-IoT2022 CSV, validating its existence,
and printing a comprehensive summary of the dataset.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.config import DATASET_PATH, TARGET_COLUMN


def load_dataset(filepath: Path | None = None) -> pd.DataFrame:
    """Load the RT-IoT2022 dataset from a CSV file.

    Parameters
    ----------
    filepath : Path, optional
        Path to the CSV.  Defaults to the configured DATASET_PATH.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    """
    filepath = filepath or DATASET_PATH

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Please download the RT-IoT2022 dataset from:\n"
            "  https://archive.ics.uci.edu/dataset/942/rt-iot2022\n"
            f"and place the CSV file at: {filepath}"
        )

    print(f"Loading dataset from {filepath} ...")
    df = pd.read_csv(filepath)
    print(f"  ✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Print and return a comprehensive summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        Dictionary containing summary statistics.
    """
    summary = {}

    # Shape
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    summary["shape"] = df.shape

    # Column names
    print(f"\nColumns ({df.shape[1]}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")
    summary["columns"] = list(df.columns)

    # Data types
    print("\nData types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    summary["dtypes"] = df.dtypes.to_dict()

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing values: {total_missing:,} total")
    if total_missing > 0:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("  No missing values found.")
    summary["missing"] = missing.to_dict()

    # Duplicates
    n_duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_duplicates:,} ({n_duplicates / len(df) * 100:.2f}%)")
    summary["duplicates"] = n_duplicates

    # Target distribution
    if TARGET_COLUMN in df.columns:
        print(f"\nTarget column: '{TARGET_COLUMN}'")
        target_dist = df[TARGET_COLUMN].value_counts()
        print(f"  Unique classes: {len(target_dist)}")
        print("\n  Class distribution:")
        for cls, count in target_dist.items():
            pct = count / len(df) * 100
            print(f"    {cls}: {count:,} ({pct:.2f}%)")
        summary["target_distribution"] = target_dist.to_dict()
    else:
        print(f"\n  ⚠ Target column '{TARGET_COLUMN}' not found in dataset!")
        print(f"    Available columns: {list(df.columns)}")

    # First few rows
    print("\nFirst 5 rows (transposed):")
    print(df.head().T.to_string())

    # Basic statistics for numerical columns
    numerical = df.select_dtypes(include=["number"])
    if not numerical.empty:
        print(f"\nNumerical features: {numerical.shape[1]}")
        print(numerical.describe().T[["mean", "std", "min", "max"]].to_string())

    print("\n" + "=" * 60)

    return summary
