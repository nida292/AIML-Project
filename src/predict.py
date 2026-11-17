"""
Prediction module.

Loads the saved pipeline and generates predictions on new data.
Supports both programmatic use and CSV-based batch prediction.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    METADATA_SAVE_PATH,
    MODEL_SAVE_PATH,
    PREDICTIONS_DIR,
)
from src.utils import create_output_directories, map_to_binary, save_dataframe


def load_model(model_path: Path | None = None) -> tuple:
    """Load the saved pipeline and its metadata.

    Returns
    -------
    pipeline : sklearn Pipeline
    metadata : dict
    """
    model_path = model_path or MODEL_SAVE_PATH
    metadata_path = model_path.parent / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run 'python src/train.py' first to train and save a model."
        )

    pipeline = joblib.load(model_path)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return pipeline, metadata


def validate_input(
    df: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Validate that input data has the expected feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        User-provided data.
    metadata : dict
        Model metadata with feature lists.

    Returns
    -------
    pd.DataFrame
        Validated (and possibly reordered) DataFrame.
    """
    expected_numerical = metadata.get("features_numerical", [])
    expected_categorical = metadata.get("features_categorical", [])
    expected_features = expected_numerical + expected_categorical

    if not expected_features:
        print("  ⚠ No feature list in metadata; using all columns.")
        return df

    # Check for missing columns
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing {len(missing)} required feature(s):\n"
            f"  {missing[:20]}"
            + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
        )

    # Select and reorder to match training order
    return df[expected_features]


def predict(
    df: pd.DataFrame,
    pipeline=None,
    metadata: dict | None = None,
) -> pd.DataFrame:
    """Generate predictions for input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input network-flow data.
    pipeline : sklearn Pipeline, optional
        Pre-loaded pipeline. If None, loads from disk.
    metadata : dict, optional
        Pre-loaded metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Predicted_Attack_Type
        - Traffic_Status (Normal / Malicious)
        - Prediction_Probability (if model supports predict_proba)
    """
    if pipeline is None:
        pipeline, metadata = load_model()
    if metadata is None:
        metadata = {}

    # Validate and reorder
    X = validate_input(df, metadata)

    # Replace infinities
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Predict
    predictions = pipeline.predict(X)
    results = pd.DataFrame({"Predicted_Attack_Type": predictions})

    # Binary interpretation
    results["Traffic_Status"] = map_to_binary(results["Predicted_Attack_Type"])

    # Probabilities (if available)
    if hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(X)
            max_proba = np.max(probas, axis=1)
            results["Prediction_Probability"] = np.round(max_proba, 4)
        except Exception:
            # Some models don't support predict_proba reliably
            pass

    return results


def predict_from_csv(
    input_path: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Load a CSV, predict, and save results.

    Parameters
    ----------
    input_path : Path
        Path to CSV file with network-flow features.
    output_path : Path, optional
        Where to save predictions. Defaults to outputs/predictions/.

    Returns
    -------
    pd.DataFrame
    """
    create_output_directories()

    print(f"Loading input data from {input_path} ...")
    input_df = pd.read_csv(input_path)
    print(f"  ✓ Loaded {len(input_df):,} records")

    pipeline, metadata = load_model()
    results = predict(input_df, pipeline, metadata)

    # Combine input with predictions
    output_df = pd.concat([input_df.reset_index(drop=True),
                           results.reset_index(drop=True)], axis=1)

    # Save
    if output_path is None:
        output_path = PREDICTIONS_DIR / "predictions.csv"
    save_dataframe(output_df, output_path, "predictions")

    # Summary
    print(f"\n── Prediction Summary ──")
    print(f"  Total records:  {len(results):,}")

    status_counts = results["Traffic_Status"].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count:,}")

    attack_counts = results["Predicted_Attack_Type"].value_counts()
    print(f"\n  Attack type distribution:")
    for attack, count in attack_counts.items():
        print(f"    {attack}: {count:,}")

    return output_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <input_csv> [output_csv]")
        print("Example: python src/predict.py data/new_traffic.csv")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    predict_from_csv(input_file, output_file)
