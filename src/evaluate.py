"""
Model evaluation module.

Provides functions for:
  - confusion-matrix visualisation
  - binary (Normal vs Malicious) evaluation
  - feature-importance plots
  - standalone re-evaluation of a saved model
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_SAVE_PATH,
    TARGET_COLUMN,
)
from src.data_loader import load_dataset
from src.preprocessing import (
    get_feature_names,
    identify_columns,
    prepare_features_target,
    split_data,
)
from src.utils import create_output_directories, map_to_binary, save_dataframe


# ──────────────────────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
    filename: str = "confusion_matrix.png",
) -> None:
    """Generate and save a confusion-matrix heatmap."""
    print("\n  Plotting confusion matrix ...")
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Multi-class Classification", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)
    print(f"    ✓ Saved {output_dir / filename}")


# ──────────────────────────────────────────────────────────────
# Binary evaluation (Normal vs Malicious)
# ──────────────────────────────────────────────────────────────

def binary_evaluation(
    y_true: pd.Series,
    y_pred: np.ndarray,
    figures_dir: Path = FIGURES_DIR,
    metrics_dir: Path = METRICS_DIR,
) -> dict:
    """Evaluate predictions in a binary Normal / Malicious frame.

    The primary task remains multi-class; this is a secondary
    interpretation that groups normal-traffic classes together.
    """
    print("\n" + "=" * 60)
    print("BINARY EVALUATION (Normal vs Malicious)")
    print("=" * 60)

    y_true_bin = map_to_binary(y_true)
    y_pred_bin = map_to_binary(pd.Series(y_pred))

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, pos_label="Malicious",
                           zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, pos_label="Malicious",
                       zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, pos_label="Malicious",
                  zero_division=0)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # Classification report
    report_str = classification_report(y_true_bin, y_pred_bin, zero_division=0)
    print(f"\n{report_str}")

    # Save binary metrics
    binary_metrics = pd.DataFrame([{
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
    }])
    save_dataframe(binary_metrics, metrics_dir / "binary_evaluation.csv",
                   "binary evaluation")

    # Binary confusion matrix
    labels_bin = ["Normal", "Malicious"]
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=labels_bin)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_bin, annot=True, fmt="d", cmap="Oranges",
        xticklabels=labels_bin, yticklabels=labels_bin, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Binary (Normal vs Malicious)")
    plt.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix_binary.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved binary confusion matrix")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ──────────────────────────────────────────────────────────────
# Feature importance
# ──────────────────────────────────────────────────────────────

def plot_feature_importance(
    pipeline,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Extract and plot feature importances from a tree-based pipeline."""
    print("\n── Feature Importance ──")

    classifier = pipeline.named_steps["classifier"]

    # Check if the classifier has feature_importances_
    if not hasattr(classifier, "feature_importances_"):
        print("  ⚠ Classifier does not support feature_importances_. Skipping.")
        return

    importances = classifier.feature_importances_
    feature_names = get_feature_names(pipeline)

    # Build DataFrame
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Print top 20
    print("\n  Top 20 Features:")
    print(fi_df.head(20).to_string(index=False))

    # Save full feature importance
    save_dataframe(fi_df, output_dir.parent / "metrics" / "feature_importance.csv",
                   "feature importance")

    # ── Top-20 plot ──
    top_n = min(20, len(fi_df))
    top = fi_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top["Feature"].values[::-1],
        top["Importance"].values[::-1],
        color=sns.color_palette("viridis", top_n),
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Most Important Features")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {output_dir / 'feature_importance.png'}")

    # ── Top-10 plot ──
    top10 = fi_df.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top10["Feature"].values[::-1],
        top10["Importance"].values[::-1],
        color=sns.color_palette("magma", 10),
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 10 Most Important Features")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance_top10.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {output_dir / 'feature_importance_top10.png'}")


# ──────────────────────────────────────────────────────────────
# Standalone evaluation (re-evaluate a saved model)
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """Re-load the saved model and run full evaluation on the test split.

    Because the split uses a fixed random_state, the test set is
    identical to the one used during training.
    """
    create_output_directories()

    print("=" * 60)
    print("RE-EVALUATING SAVED MODEL")
    print("=" * 60)

    # Load model
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(
            f"No saved model found at {MODEL_SAVE_PATH}.\n"
            "Run 'python src/train.py' first."
        )
    pipeline = joblib.load(MODEL_SAVE_PATH)
    print(f"  ✓ Loaded model from {MODEL_SAVE_PATH}")

    # Reconstruct the same test split
    df = load_dataset()
    columns_info = identify_columns(df)
    X, y = prepare_features_target(df, columns_info)
    _, X_test, _, y_test = split_data(X, y)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Multi-class evaluation
    print("\n── Multi-class Classification Report ──\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR)

    # Binary evaluation
    binary_evaluation(y_test, y_pred)

    # Feature importance
    plot_feature_importance(pipeline)

    print("\n  ✓ Evaluation complete.")


if __name__ == "__main__":
    main()
