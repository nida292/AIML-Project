"""
Model training pipeline.

Orchestrates the end-to-end workflow:
  load → inspect → EDA → split → train → compare → tune → evaluate → save

Run:
    python src/train.py
"""

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DT_PARAM_GRID,
    FIGURES_DIR,
    HIST_GB_PARAM_GRID,
    LR_PARAM_GRID,
    METRICS_DIR,
    MODEL_SAVE_PATH,
    METADATA_SAVE_PATH,
    RANDOM_STATE,
    RF_PARAM_GRID,
    TARGET_COLUMN,
)
from src.data_loader import inspect_dataset, load_dataset
from src.eda import run_full_eda
from src.evaluate import (
    binary_evaluation,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.preprocessing import (
    build_pipeline,
    get_feature_names,
    identify_columns,
    prepare_features_target,
    split_data,
)
from src.utils import create_output_directories, format_duration, save_dataframe

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dictionary of model name → classifier instance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


# ──────────────────────────────────────────────────────────────
# Training & comparison
# ──────────────────────────────────────────────────────────────

def train_and_compare(
    models: dict,
    numerical_cols: list,
    categorical_cols: list,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Train every model, evaluate on the test set, return comparison table.

    Each model gets its own fresh preprocessing pipeline to avoid
    shared-state issues.
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING & COMPARISON")
    print("=" * 60)

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        print(f"\n── {name} ──")
        pipeline = build_pipeline(numerical_cols, categorical_cols, clf)

        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (weighted)": round(prec, 4),
            "Recall (weighted)": round(rec, 4),
            "F1 (weighted)": round(f1_w, 4),
            "F1 (macro)": round(f1_m, 4),
            "Training Time": format_duration(train_time),
        })
        trained_pipelines[name] = pipeline

        print(f"  Accuracy:      {acc:.4f}")
        print(f"  Precision (w): {prec:.4f}")
        print(f"  Recall (w):    {rec:.4f}")
        print(f"  F1 weighted:   {f1_w:.4f}")
        print(f"  F1 macro:      {f1_m:.4f}")
        print(f"  Time:          {format_duration(train_time)}")

    comparison_df = pd.DataFrame(results)
    print("\n── Model Comparison ──")
    print(comparison_df.to_string(index=False))

    return comparison_df, trained_pipelines


# ──────────────────────────────────────────────────────────────
# Hyperparameter tuning
# ──────────────────────────────────────────────────────────────

def tune_model(
    name: str,
    pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    cv: int = 3,
) -> object:
    """Run RandomizedSearchCV on a pipeline.

    Parameters
    ----------
    name : str
        Model name (for display).
    pipeline : Pipeline
    param_grid : dict
    X_train, y_train : training data
    n_iter : int
        Number of random combinations to try.
    cv : int
        Cross-validation folds.

    Returns
    -------
    best_pipeline : Pipeline
        Refitted on the full training set with best params.
    """
    print(f"\n── Tuning {name} ──")
    print(f"  Search space: {n_iter} iterations × {cv}-fold CV")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"  Best F1 macro (CV): {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    print(f"  Tuning time: {format_duration(elapsed)}")

    return search.best_estimator_


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run the complete training pipeline."""
    create_output_directories()

    # 1 ── Load & inspect ──────────────────────────────────────
    df = load_dataset()
    summary = inspect_dataset(df)

    # 2 ── EDA ─────────────────────────────────────────────────
    run_full_eda(df)

    # 3 ── Identify columns ────────────────────────────────────
    columns_info = identify_columns(df)
    numerical_cols = columns_info["numerical"]
    categorical_cols = columns_info["categorical"]

    # 4 ── Features & target ───────────────────────────────────
    X, y = prepare_features_target(df, columns_info)

    # 5 ── Train / test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 6 ── Train & compare models ──────────────────────────────
    models = get_models()
    comparison_df, trained_pipelines = train_and_compare(
        models, numerical_cols, categorical_cols,
        X_train, X_test, y_train, y_test,
    )
    save_dataframe(comparison_df, METRICS_DIR / "model_comparison.csv",
                   "model comparison")

    # 7 ── Select best model by macro F1 ───────────────────────
    best_model_name = comparison_df.sort_values(
        "F1 (macro)", ascending=False
    ).iloc[0]["Model"]
    print(f"\n  ★ Best model (by macro F1): {best_model_name}")

    # 8 ── Hyperparameter tuning on top candidates ─────────────
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    # Determine which grids to use for tuning
    tuning_candidates = {
        "Random Forest": RF_PARAM_GRID,
        "HistGradientBoosting": HIST_GB_PARAM_GRID,
        "Decision Tree": DT_PARAM_GRID,
        "Logistic Regression": LR_PARAM_GRID,
    }

    # Tune the top 2 models by macro F1
    top_models = comparison_df.sort_values(
        "F1 (macro)", ascending=False
    ).head(2)["Model"].tolist()

    tuned_pipelines = {}
    for model_name in top_models:
        if model_name in tuning_candidates:
            pipeline = trained_pipelines[model_name]
            # Build a fresh pipeline for tuning
            clf = get_models()[model_name]
            fresh_pipeline = build_pipeline(numerical_cols, categorical_cols, clf)
            tuned = tune_model(
                model_name, fresh_pipeline,
                tuning_candidates[model_name],
                X_train, y_train,
                n_iter=10, cv=3,
            )
            tuned_pipelines[model_name] = tuned
        else:
            print(f"\n  No tuning grid for {model_name}; using default.")
            tuned_pipelines[model_name] = trained_pipelines[model_name]

    # 9 ── Evaluate tuned models and select final ──────────────
    print("\n" + "=" * 60)
    print("TUNED MODEL EVALUATION")
    print("=" * 60)

    best_f1 = -1
    final_pipeline = None
    final_model_name = None

    for name, pipeline in tuned_pipelines.items():
        y_pred = pipeline.predict(X_test)
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n  {name}:")
        print(f"    Accuracy:    {acc:.4f}")
        print(f"    F1 macro:    {f1_m:.4f}")
        print(f"    F1 weighted: {f1_w:.4f}")
        if f1_m > best_f1:
            best_f1 = f1_m
            final_pipeline = pipeline
            final_model_name = name

    # Also compare with the original (untuned) best model
    untuned_best = trained_pipelines[best_model_name]
    y_pred_untuned = untuned_best.predict(X_test)
    f1_untuned = f1_score(y_test, y_pred_untuned, average="macro", zero_division=0)
    if f1_untuned > best_f1:
        final_pipeline = untuned_best
        final_model_name = best_model_name
        best_f1 = f1_untuned
        print(f"\n  ★ Untuned {best_model_name} still outperforms tuned versions.")

    print(f"\n  ★ Final model selected: {final_model_name} (macro F1 = {best_f1:.4f})")

    # 10 ── Full evaluation of final model ─────────────────────
    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    y_pred_final = final_pipeline.predict(X_test)

    # Classification report
    report_str = classification_report(y_test, y_pred_final, zero_division=0)
    print("\n  Classification Report:\n")
    print(report_str)

    # Save classification report as CSV
    report_dict = classification_report(
        y_test, y_pred_final, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T
    save_dataframe(report_df, METRICS_DIR / "final_classification_report.csv",
                   "classification report")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred_final, FIGURES_DIR)

    # 11 ── Binary evaluation ──────────────────────────────────
    binary_evaluation(y_test, y_pred_final, FIGURES_DIR, METRICS_DIR)

    # 12 ── Feature importance ─────────────────────────────────
    plot_feature_importance(final_pipeline, FIGURES_DIR)

    # 13 ── Save model & metadata ──────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_SAVE_PATH)
    print(f"  ✓ Model saved to {MODEL_SAVE_PATH}")

    metadata = {
        "model_name": final_model_name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset": "RT-IoT2022",
        "target_column": TARGET_COLUMN,
        "features_numerical": numerical_cols,
        "features_categorical": categorical_cols,
        "n_features_total": len(numerical_cols) + len(categorical_cols),
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "macro_f1": round(best_f1, 4),
        "model_version": "1.0",
    }
    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to {METADATA_SAVE_PATH}")

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final model:    {final_model_name}")
    print(f"  Macro F1:       {best_f1:.4f}")
    print(f"  Model file:     {MODEL_SAVE_PATH}")
    print(f"  Metadata:       {METADATA_SAVE_PATH}")
    print(f"  Figures:        {FIGURES_DIR}")
    print(f"  Metrics:        {METRICS_DIR}")
    print()


if __name__ == "__main__":
    main()
