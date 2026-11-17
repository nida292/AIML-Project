"""
Exploratory Data Analysis module.

Generates cybersecurity-oriented visualisations for the RT-IoT2022
dataset and saves them to the configured figures directory.
"""

import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, NORMAL_CATEGORIES, TARGET_COLUMN
from src.utils import map_to_binary


def run_full_eda(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    normal_categories: List[str] | None = None,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Run the complete EDA pipeline and save all figures.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    target_col : str
        Name of the target column.
    normal_categories : list[str], optional
        Categories considered as normal traffic.
    output_dir : Path
        Directory to save figures.
    """
    if normal_categories is None:
        normal_categories = NORMAL_CATEGORIES
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Class distribution
    plot_class_distribution(df, target_col, output_dir)

    # 2. Binary distribution (Normal vs Malicious)
    plot_binary_distribution(df, target_col, normal_categories, output_dir)

    # 3. Class imbalance ratio
    plot_class_imbalance(df, target_col, output_dir)

    # 4. Numerical feature distributions
    plot_numerical_distributions(df, target_col, output_dir)

    # 5. Correlation heatmap
    plot_correlation_matrix(df, output_dir)

    # 6. Boxplots for selected features
    plot_boxplots(df, target_col, output_dir)

    # 7. Missing-value summary (visual)
    plot_missing_values(df, output_dir)

    print(f"\n  ✓ All EDA figures saved to {output_dir}")


# ──────────────────────────────────────────────────────────────
# Individual plot functions
# ──────────────────────────────────────────────────────────────

def plot_class_distribution(
    df: pd.DataFrame, target_col: str, output_dir: Path
) -> None:
    """Bar chart of per-class sample counts."""
    print("\n  Plotting class distribution ...")
    class_counts = df[target_col].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(class_counts.index.astype(str), class_counts.values,
                   color=sns.color_palette("viridis", len(class_counts)))
    ax.set_xlabel("Number of Samples")
    ax.set_title("Attack Type Distribution in RT-IoT2022 Dataset")
    ax.invert_yaxis()

    # Annotate counts
    for bar, count in zip(bars, class_counts.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_binary_distribution(
    df: pd.DataFrame,
    target_col: str,
    normal_categories: List[str],
    output_dir: Path,
) -> None:
    """Pie chart of Normal vs Malicious traffic."""
    print("  Plotting binary distribution ...")
    binary_labels = map_to_binary(df[target_col], normal_categories)
    counts = binary_labels.value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    colours = ["#2ecc71", "#e74c3c"]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colours,
        startangle=140,
        textprops={"fontsize": 12},
    )
    ax.set_title("Normal vs Malicious Traffic", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "binary_distribution.png", dpi=150)
    plt.close(fig)


def plot_class_imbalance(
    df: pd.DataFrame, target_col: str, output_dir: Path
) -> None:
    """Bar chart showing imbalance ratios relative to the largest class."""
    print("  Plotting class imbalance ratios ...")
    counts = df[target_col].value_counts()
    ratios = counts / counts.max()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(ratios.index.astype(str), ratios.values,
                   color=sns.color_palette("rocket", len(ratios)))
    ax.set_xlabel("Ratio (relative to largest class)")
    ax.set_title("Class Imbalance Ratios")
    ax.invert_yaxis()

    for bar, ratio in zip(bars, ratios.values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ratio:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "class_imbalance.png", dpi=150)
    plt.close(fig)


def plot_numerical_distributions(
    df: pd.DataFrame, target_col: str, output_dir: Path,
    max_features: int = 12,
) -> None:
    """Histograms for selected numerical features."""
    print("  Plotting numerical feature distributions ...")
    numerical = df.select_dtypes(include=[np.number])
    if numerical.empty:
        print("    No numerical features found.")
        return

    # Select features with highest variance
    variances = numerical.var().sort_values(ascending=False)
    selected = variances.head(max_features).index.tolist()

    n_cols = 3
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, col in enumerate(selected):
        ax = axes[i]
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(data, bins=50, edgecolor="black", alpha=0.7, color="#3498db")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    # Hide unused axes
    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions (Top by Variance)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "numerical_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(
    df: pd.DataFrame, output_dir: Path, max_features: int = 20
) -> None:
    """Heatmap of Pearson correlations for numerical features."""
    print("  Plotting correlation matrix ...")
    numerical = df.select_dtypes(include=[np.number])
    if numerical.shape[1] < 2:
        print("    Not enough numerical features for correlation.")
        return

    # Use top features by variance to keep the heatmap readable
    variances = numerical.var().sort_values(ascending=False)
    selected = variances.head(max_features).index.tolist()
    corr = numerical[selected].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="coolwarm", center=0,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Top by Variance)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_boxplots(
    df: pd.DataFrame, target_col: str, output_dir: Path,
    max_features: int = 6,
) -> None:
    """Boxplots of selected numerical features grouped by attack type."""
    print("  Plotting boxplots ...")
    numerical = df.select_dtypes(include=[np.number])
    if numerical.empty:
        return

    # Select features with highest variance
    variances = numerical.var().sort_values(ascending=False)
    selected = variances.head(max_features).index.tolist()

    n_cols = 2
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, col in enumerate(selected):
        ax = axes[i]
        plot_df = df[[col, target_col]].copy()
        plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
        sns.boxplot(data=plot_df, x=target_col, y=col, ax=ax,
                    palette="Set2", showfliers=False)
        ax.set_title(col, fontsize=11)
        ax.tick_params(axis="x", rotation=45)

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Boxplots by Attack Type", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of missing values per column (only columns with missing)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("  No missing values to plot.")
        return

    print("  Plotting missing values ...")
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    ax.barh(missing.index.astype(str), missing.values, color="#e67e22")
    ax.set_xlabel("Number of Missing Values")
    ax.set_title("Missing Values per Feature")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / "missing_values.png", dpi=150)
    plt.close(fig)
