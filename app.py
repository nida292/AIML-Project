"""
Streamlit Web Application
Machine Learning-Based Network Intrusion Detection

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Resolve project root so imports work when streamlit is launched
#    from the project directory.
import sys
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    METADATA_SAVE_PATH,
    MODEL_SAVE_PATH,
)
from src.predict import load_model, predict


# ───────────────────────── Page config ─────────────────────────
st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)


# ───────────────────────── Sidebar ─────────────────────────────
st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "About the Project",
        "Dataset Information",
        "Upload & Predict",
        "Model Performance",
        "Feature Importance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**ML-Based Network Intrusion Detection**\n\n"
    "MSc Cybersecurity Project\n\n"
    "Dataset: RT-IoT2022"
)


# ─────────────────────── Helper loaders ────────────────────────

@st.cache_resource
def cached_load_model():
    """Load the model once and cache it."""
    try:
        return load_model()
    except FileNotFoundError:
        return None, None


def load_metadata() -> dict:
    """Load model metadata JSON."""
    if METADATA_SAVE_PATH.exists():
        with open(METADATA_SAVE_PATH) as f:
            return json.load(f)
    return {}


def load_image(filename: str):
    """Load a figure from the figures directory."""
    path = FIGURES_DIR / filename
    if path.exists():
        return str(path)
    return None


@st.cache_data
def load_dataset_for_sampling():
    """Load RT-IoT2022 dataset once for fast dynamic sampling."""
    candidates = [
        PROJECT_ROOT / "data" / "RT_IoT2022.csv",
        PROJECT_ROOT / "RT_IoT2022.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def generate_random_sample(n_per_class: int = 4) -> pd.DataFrame | None:
    """Dynamically sample n records from each class in RT_IoT2022."""
    df_full = load_dataset_for_sampling()
    if df_full is None:
        sample_fallback = PROJECT_ROOT / "data" / "sample_traffic.csv"
        if sample_fallback.exists():
            return pd.read_csv(sample_fallback)
        return None

    target_col = "Attack_type" if "Attack_type" in df_full.columns else None
    if not target_col:
        return df_full.sample(n=min(48, len(df_full)))

    samples = []
    for _, group in df_full.groupby(target_col):
        n = min(n_per_class, len(group))
        samples.append(group.sample(n=n))

    sampled = pd.concat(samples, ignore_index=True).sample(frac=1).reset_index(drop=True)
    drop_cols = [c for c in ["Unnamed: 0", target_col] if c in sampled.columns]
    return sampled.drop(columns=drop_cols)


# ═══════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════

# ──────────────── 1. About the Project ─────────────────────────
if page == "About the Project":
    st.title("🛡️ Machine Learning-Based Network Intrusion Detection")
    st.markdown(
        """
        ### Project Overview

        This system uses **supervised machine learning** to analyse network-flow
        traffic and classify it into **normal traffic** or **specific attack
        categories**.

        It was built as part of an **MSc Cybersecurity** project to demonstrate
        how machine-learning techniques can enhance network intrusion detection.

        ---

        ### How It Works

        1. **Network traffic features** (flow statistics, protocol info, etc.)
           are extracted from network captures.
        2. The trained ML model analyses these features.
        3. Each record is classified as a **specific attack type** or
           **normal traffic**.
        4. A secondary **binary interpretation** (Normal vs Malicious) is
           also provided.

        ---

        ### Key Features

        | Feature | Description |
        |---------|-------------|
        | Multi-class classification | Identifies specific attack types |
        | Binary classification | Normal vs Malicious |
        | Batch prediction | Upload CSV files for bulk analysis |
        | Model explainability | Feature importance visualisation |
        | Performance metrics | Comprehensive evaluation reports |

        ---

        ### Attack Categories Detected

        - DOS_SYN_Hping
        - ARP_poisioning
        - NMAP_UDP_SCAN
        - NMAP_XMAS_TREE_SCAN
        - NMAP_OS_DETECTION
        - NMAP_TCP_scan
        - DDOS_Slowloris
        - Metasploit_Brute_Force_SSH
        - NMAP_FIN_SCAN

        ### Normal Traffic Categories

        - MQTT
        - Thing_speak
        - Wipro_bulb_Dataset
        - Amazon-Alexa
        """
    )

# ──────────────── 2. Dataset Information ───────────────────────
elif page == "Dataset Information":
    st.title("📊 Dataset Information")

    st.markdown(
        """
        ### RT-IoT2022 Dataset

        | Property | Value |
        |----------|-------|
        | **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/942/rt-iot2022) |
        | **Records** | ~123,117 |
        | **Type** | Network flow / traffic features |
        | **Target** | `Attack_type` (multi-class) |
        | **Domain** | IoT network security |

        The dataset contains a mixture of **normal IoT traffic** and various
        **network attack** scenarios captured in a controlled environment.
        """
    )

    # Show class distribution if figure exists
    dist_img = load_image("class_distribution.png")
    if dist_img:
        st.subheader("Class Distribution")
        st.image(dist_img, use_container_width=True)

    bin_img = load_image("binary_distribution.png")
    if bin_img:
        st.subheader("Normal vs Malicious Distribution")
        st.image(bin_img, use_container_width=True)

    imb_img = load_image("class_imbalance.png")
    if imb_img:
        st.subheader("Class Imbalance Ratios")
        st.image(imb_img, use_container_width=True)

# ──────────────── 3. Upload & Predict ──────────────────────────
elif page == "Upload & Predict":
    st.title("🔍 Upload Network Traffic & Predict")

    pipeline, metadata = cached_load_model()

    if pipeline is None:
        st.error(
            "⚠️ No trained model found. Please run `python src/train.py` "
            "first to train and save a model."
        )
    else:
        st.success("✅ Model loaded successfully!")
        meta = load_metadata()
        if meta:
            st.info(
                f"**Model:** {meta.get('model_name', 'N/A')} · "
                f"**Macro F1:** {meta.get('macro_f1', 'N/A')} · "
                f"**Trained:** {meta.get('training_date', 'N/A')[:10]}"
            )

        st.markdown(
            """
            ### Instructions

            1. Upload a CSV file containing **network-flow features** below, OR
            2. Click **"🎲 Generate New Random Sample"** to draw a fresh set of 48 random network flows directly from `RT_IoT2022.csv` (across normal and attack classes).
            3. The system will classify each record and display detailed detection results.
            """
        )

        col_sample, _ = st.columns([1, 1])
        with col_sample:
            if st.button("🎲 Generate New Random Sample (48 records)"):
                with st.spinner("Extracting fresh random network flows..."):
                    st.session_state["sample_data"] = generate_random_sample(n_per_class=4)
                    st.session_state["sample_count"] = st.session_state.get("sample_count", 0) + 1

        uploaded_file = st.file_uploader(
            "Upload a CSV file with network traffic data",
            type=["csv"],
        )

        input_df = None
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
        elif "sample_data" in st.session_state and st.session_state["sample_data"] is not None:
            input_df = st.session_state["sample_data"]
            sample_num = st.session_state.get("sample_count", 1)
            st.info(f"🎲 Using freshly generated random sample #{sample_num} (48 records across 12 categories from RT-IoT2022).")

        if input_df is not None:
            try:
                st.write(f"**Loaded:** {len(input_df):,} records × "
                         f"{input_df.shape[1]} columns")

                with st.expander("Preview uploaded data"):
                    st.dataframe(input_df.head(20))

                # Predict
                with st.spinner("Classifying network traffic..."):
                    results = predict(input_df, pipeline, metadata)

                st.markdown("---")
                st.subheader("📋 Prediction Results")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(results):,}")
                with col2:
                    n_normal = (results["Traffic_Status"] == "Normal").sum()
                    st.metric("Normal", f"{n_normal:,}")
                with col3:
                    n_malicious = (results["Traffic_Status"] == "Malicious").sum()
                    st.metric("Malicious", f"{n_malicious:,}")

                # Traffic status distribution
                st.subheader("Traffic Status Distribution")
                status_counts = results["Traffic_Status"].value_counts()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.bar_chart(status_counts)
                with col_b:
                    st.dataframe(status_counts.reset_index().rename(
                        columns={"index": "Status", "Traffic_Status": "Count"}
                    ))

                # Attack type distribution
                st.subheader("Attack Type Distribution")
                attack_counts = results["Predicted_Attack_Type"].value_counts()
                st.bar_chart(attack_counts)

                # Full results table
                with st.expander("View all predictions"):
                    full_output = pd.concat([
                        input_df.reset_index(drop=True),
                        results.reset_index(drop=True),
                    ], axis=1)
                    st.dataframe(full_output)

                # Download
                st.markdown("---")
                output_csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=output_csv,
                    file_name="intrusion_detection_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.markdown(
                    "Make sure your CSV contains the required feature columns. "
                    "Check the model metadata for the expected column list."
                )
                if meta:
                    with st.expander("Expected features"):
                        features = (
                            meta.get("features_numerical", [])
                            + meta.get("features_categorical", [])
                        )
                        st.write(features)

# ──────────────── 4. Model Performance ─────────────────────────
elif page == "Model Performance":
    st.title("📈 Model Performance")

    # Model comparison
    comparison_path = METRICS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        st.subheader("Model Comparison")
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Model comparison data not available. Run training first.")

    # Classification report
    report_path = METRICS_DIR / "final_classification_report.csv"
    if report_path.exists():
        st.subheader("Final Model — Classification Report")
        report_df = pd.read_csv(report_path)
        st.dataframe(report_df, use_container_width=True)

    # Confusion matrix
    cm_img = load_image("confusion_matrix.png")
    if cm_img:
        st.subheader("Confusion Matrix — Multi-class")
        st.image(cm_img, use_container_width=True)

    # Binary confusion matrix
    cm_bin_img = load_image("confusion_matrix_binary.png")
    if cm_bin_img:
        st.subheader("Confusion Matrix — Binary (Normal vs Malicious)")
        st.image(cm_bin_img, use_container_width=True)

    # Binary metrics
    binary_path = METRICS_DIR / "binary_evaluation.csv"
    if binary_path.exists():
        st.subheader("Binary Evaluation Metrics")
        binary_df = pd.read_csv(binary_path)
        st.dataframe(binary_df, use_container_width=True)

# ──────────────── 5. Feature Importance ────────────────────────
elif page == "Feature Importance":
    st.title("🔑 Feature Importance")

    st.markdown(
        """
        Feature importance shows which network-flow characteristics
        contribute most to the model's classification decisions.
        Understanding important features helps explain **why** certain
        traffic is flagged as malicious.
        """
    )

    fi_img = load_image("feature_importance.png")
    if fi_img:
        st.subheader("Top 20 Most Important Features")
        st.image(fi_img, use_container_width=True)

    fi10_img = load_image("feature_importance_top10.png")
    if fi10_img:
        st.subheader("Top 10 Most Important Features")
        st.image(fi10_img, use_container_width=True)

    # Feature importance table
    fi_path = METRICS_DIR / "feature_importance.csv"
    if fi_path.exists():
        st.subheader("Feature Importance Table")
        fi_df = pd.read_csv(fi_path)
        st.dataframe(fi_df, use_container_width=True)

    if not fi_img and not fi_path.exists():
        st.info(
            "Feature importance data not available. "
            "Run training first with `python src/train.py`."
        )
