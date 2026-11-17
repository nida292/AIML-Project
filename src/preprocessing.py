"""
Data preprocessing module.

Builds a scikit-learn Pipeline + ColumnTransformer that handles
categorical encoding and numerical scaling, fitted only on
the training split to prevent data leakage.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ID_COLUMN_PATTERNS,
    MAX_ONEHOT_CARDINALITY,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils import clean_infinite_values


def identify_columns(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> Dict[str, List[str]]:
    """Programmatically identify column roles.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    target_col : str
        Name of the target column.

    Returns
    -------
    dict
        Keys: 'target', 'id_columns', 'numerical', 'categorical',
              'high_cardinality', 'all_features'.
    """
    columns_info: Dict[str, List[str]] = {
        "target": [target_col],
        "id_columns": [],
        "numerical": [],
        "categorical": [],
        "high_cardinality": [],
        "all_features": [],
    }

    for col in df.columns:
        # Skip target
        if col == target_col:
            continue

        # Detect ID-like columns (exact match on lowered name or
        # "unnamed" at the start, or a column named just "id")
        col_lower = col.lower().strip()
        if col_lower in ID_COLUMN_PATTERNS or col_lower.startswith("unnamed"):
            columns_info["id_columns"].append(col)
            continue

        # Classify remaining columns by dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            columns_info["numerical"].append(col)
        else:
            # Check cardinality for categorical columns
            n_unique = df[col].nunique()
            if n_unique > MAX_ONEHOT_CARDINALITY:
                columns_info["high_cardinality"].append(col)
                print(
                    f"  ⚠ Dropping high-cardinality categorical column "
                    f"'{col}' ({n_unique} unique values > {MAX_ONEHOT_CARDINALITY})"
                )
            else:
                columns_info["categorical"].append(col)

    columns_info["all_features"] = (
        columns_info["numerical"] + columns_info["categorical"]
    )

    # Summary
    print("\n── Column identification ──")
    print(f"  Target:             {columns_info['target']}")
    print(f"  ID columns dropped: {columns_info['id_columns']}")
    print(f"  Numerical features: {len(columns_info['numerical'])}")
    print(f"  Categorical features: {len(columns_info['categorical'])}")
    if columns_info["high_cardinality"]:
        print(f"  High-cardinality (dropped): {columns_info['high_cardinality']}")
    print(f"  Total features used: {len(columns_info['all_features'])}")

    return columns_info


def prepare_features_target(
    df: pd.DataFrame,
    columns_info: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y), dropping ID / high-card columns.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    columns_info : dict
        Output of :func:`identify_columns`.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    target_col = columns_info["target"][0]
    drop_cols = (
        columns_info["target"]
        + columns_info["id_columns"]
        + columns_info["high_cardinality"]
    )
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col].copy()

    # Clean infinite values in numerical columns
    X = clean_infinite_values(X)

    print(f"\n  Features (X): {X.shape}")
    print(f"  Target  (y): {y.shape} — {y.nunique()} classes")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"\n── Train / Test Split ──")
    print(f"  Training set:  {X_train.shape[0]:,} samples ({1 - test_size:.0%})")
    print(f"  Test set:      {X_test.shape[0]:,} samples ({test_size:.0%})")
    return X_train, X_test, y_train, y_test


def create_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for numerical scaling and categorical encoding.

    Parameters
    ----------
    numerical_cols : list[str]
    categorical_cols : list[str]

    Returns
    -------
    ColumnTransformer
    """
    transformers = []

    if numerical_cols:
        transformers.append(
            ("num", StandardScaler(), numerical_cols)
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # drop any unlisted columns
    )
    return preprocessor


def build_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    classifier,
) -> Pipeline:
    """Create a complete Pipeline: preprocessor → classifier.

    Parameters
    ----------
    numerical_cols : list[str]
    categorical_cols : list[str]
    classifier : sklearn estimator

    Returns
    -------
    Pipeline
    """
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def get_feature_names(pipeline: Pipeline) -> list:
    """Retrieve transformed feature names from a fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted pipeline whose first step is a ColumnTransformer.

    Returns
    -------
    list[str]
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        return [f"feature_{i}" for i in range(preprocessor.n_features_in_)]
