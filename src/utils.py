"""
Utility helpers shared across modules.
"""

import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NORMAL_CATEGORIES,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
)


def create_output_directories() -> None:
    """Create all required output directories if they do not exist."""
    for directory in [MODELS_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def map_to_binary(
    labels: pd.Series,
    normal_categories: List[str] | None = None,
) -> pd.Series:
    """Map multi-class Attack_type labels to binary Normal / Malicious.

    Parameters
    ----------
    labels : pd.Series
        Series of Attack_type values.
    normal_categories : list[str], optional
        Categories considered normal traffic. Defaults to config list.

    Returns
    -------
    pd.Series
        Binary labels: "Normal" or "Malicious".
    """
    if normal_categories is None:
        normal_categories = NORMAL_CATEGORIES
    return labels.apply(
        lambda x: "Normal" if x in normal_categories else "Malicious"
    )


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"


def save_dataframe(df: pd.DataFrame, path: Path, description: str = "") -> None:
    """Save a DataFrame to CSV with a confirmation message."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  ✓ Saved {description}: {path}")


def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values with NaN so they can be handled downstream."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df
