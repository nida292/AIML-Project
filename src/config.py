"""
Configuration module for the Network Intrusion Detection System.

Centralises all paths, constants, and hyperparameter grids so that
every other module imports from a single source of truth.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

DATASET_FILENAME = "RT_IoT2022.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

MODEL_SAVE_PATH = MODELS_DIR / "intrusion_detection_model.joblib"
METADATA_SAVE_PATH = MODELS_DIR / "model_metadata.json"

# ──────────────────────────────────────────────
# ML configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Attack_type"

# Columns that must never be used as predictive features.
# Detected programmatically as well, but listed here as a safety net.
ID_COLUMN_PATTERNS = ["id", "unnamed"]

# Maximum unique values for a categorical column to be one-hot encoded.
# Columns exceeding this threshold are dropped with a warning.
MAX_ONEHOT_CARDINALITY = 50

# ──────────────────────────────────────────────
# Normal-traffic categories
# ──────────────────────────────────────────────
NORMAL_CATEGORIES = [
    "MQTT",
    "MQTT_Publish",
    "Thing_speak",
    "Thing_Speak",
    "Wipro_bulb",
    "Wipro_bulb_Dataset",
    "Amazon-Alexa",
]

# ──────────────────────────────────────────────
# Hyperparameter search grids (RandomizedSearchCV)
# ──────────────────────────────────────────────
RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, 30, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": ["balanced", "balanced_subsample"],
}

HIST_GB_PARAM_GRID = {
    "classifier__max_iter": [100, 200, 300],
    "classifier__max_depth": [5, 10, 20, None],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__min_samples_leaf": [5, 10, 20],
}

DT_PARAM_GRID = {
    "classifier__max_depth": [5, 10, 20, 30, None],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
    "classifier__class_weight": ["balanced", None],
}

LR_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__solver": ["lbfgs", "saga"],
    "classifier__class_weight": ["balanced", None],
}
