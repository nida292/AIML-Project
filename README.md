# Machine Learning-Based Network Intrusion Detection for Classifying Malicious Network Traffic

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Objectives](#objectives)
5. [Dataset Description](#dataset-description)
6. [Dataset Source](#dataset-source)
7. [Attack Categories](#attack-categories)
8. [Normal Traffic Categories](#normal-traffic-categories)
9. [ML Methodology](#ml-methodology)
10. [Data Preprocessing](#data-preprocessing)
11. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
12. [Class Imbalance Handling](#class-imbalance-handling)
13. [Models Used](#models-used)
14. [Hyperparameter Tuning](#hyperparameter-tuning)
15. [Evaluation Metrics](#evaluation-metrics)
16. [Final Model Selection](#final-model-selection)
17. [Results](#results)
18. [Feature Importance](#feature-importance)
19. [Prediction System](#prediction-system)
20. [Streamlit Application](#streamlit-application)
21. [Project Structure](#project-structure)
22. [Installation](#installation)
23. [Dataset Setup](#dataset-setup)
24. [How to Train](#how-to-train)
25. [How to Evaluate](#how-to-evaluate)
26. [How to Run the Application](#how-to-run-the-application)
27. [How to Predict from CLI](#how-to-predict-from-cli)
28. [Limitations](#limitations)
29. [Future Work](#future-work)
30. [Ethical and Security Considerations](#ethical-and-security-considerations)
31. [Dataset Citation](#dataset-citation)
32. [Viva Questions and Answers](#viva-questions-and-answers)

---

## Project Overview

This project implements a **Machine Learning-based Intrusion Detection System (IDS)** that classifies network traffic into normal or specific attack categories using supervised learning techniques. The system is trained on the **RT-IoT2022** dataset, a comprehensive real-world dataset of network flow records captured from an IoT network environment.

The project encompasses the full machine learning lifecycle — from data exploration and preprocessing through model training, hyperparameter tuning, and evaluation — culminating in an interactive **Streamlit web application** that allows users to upload network traffic data and receive predictions on whether each flow is normal or malicious, along with the specific attack type and confidence probability.

This work was completed as part of an **MSc Cybersecurity** programme, demonstrating the practical application of machine learning to a critical cybersecurity challenge.

---

## Problem Statement

Network intrusion detection is a fundamental requirement for securing modern computer networks, particularly in the rapidly expanding Internet of Things (IoT) ecosystem. Traditional signature-based Intrusion Detection Systems rely on known attack patterns and are inherently unable to detect novel or zero-day attacks. As IoT devices proliferate — often with limited computational resources and inconsistent security implementations — the attack surface grows significantly.

The core problem addressed by this project is:

> **How can supervised machine learning be applied to network flow data to accurately classify traffic as normal or belonging to specific attack categories, thereby enabling automated detection of diverse network intrusions in IoT environments?**

Manual inspection of network traffic is infeasible at scale, and rule-based systems cannot keep pace with the evolving threat landscape. There is a pressing need for intelligent, data-driven systems that can learn complex traffic patterns and generalise to detect a variety of attack types with high precision and recall.

---

## Motivation

Several factors motivate the application of machine learning to network intrusion detection:

- **Growing IoT Threat Landscape**: The number of IoT devices is projected to exceed 30 billion by 2030. These devices frequently lack robust security mechanisms, making IoT networks attractive targets for adversaries. Attacks such as DDoS, ARP poisoning, and brute-force SSH attempts are increasingly common in IoT environments.

- **Limitations of Signature-Based IDS**: Conventional IDS solutions (e.g., Snort, Suricata) rely on pattern matching against databases of known attack signatures. While effective for known threats, they fail to detect new attack variants, polymorphic attacks, or sophisticated evasion techniques. Maintaining and updating signature databases is labour-intensive and reactive.

- **Advantages of Machine Learning**: ML models learn discriminative patterns directly from data, enabling them to generalise beyond known signatures. Supervised classification models can identify subtle statistical differences between normal and malicious traffic flows, detecting both known and previously unseen attack variants. ML-based IDS can operate at scale, processing thousands of flows per second with minimal human intervention.

- **Academic and Practical Relevance**: Network intrusion detection remains an active and impactful research area in cybersecurity. Developing an ML-based IDS demonstrates the intersection of data science and security — a skill set in high demand across industry and government.

---

## Objectives

The primary objectives of this project are:

1. **Multi-class Classification**: Develop and evaluate supervised machine learning models capable of classifying network traffic into 13 distinct categories (9 attack types + 4 normal traffic types) using the RT-IoT2022 dataset.

2. **Binary Interpretation**: Derive a binary classification perspective (Normal vs. Malicious) from the multi-class predictions to provide a clear, actionable security assessment for each network flow.

3. **Feature Importance Analysis**: Identify and rank the most discriminative network flow features contributing to classification decisions, providing interpretability and security-relevant insights.

4. **Interactive Prediction Interface**: Build a user-friendly Streamlit web application that allows security analysts to upload new network traffic data (in CSV format) and receive real-time predictions with attack type labels, binary classifications, and confidence probabilities.

5. **Robust Evaluation**: Evaluate model performance using multiple metrics appropriate for imbalanced multi-class datasets, including macro F1-score, weighted F1-score, precision, recall, and confusion matrices.

6. **Reproducible Pipeline**: Implement a clean, modular, and reproducible ML pipeline with proper data splitting, preprocessing fitted only on training data, and consistent random seeds.

---

## Dataset Description

The **RT-IoT2022** dataset is a modern, publicly available benchmark dataset specifically designed for evaluating network intrusion detection systems in IoT environments.

| Property | Details |
|---|---|
| **Dataset Name** | RT-IoT2022 |
| **Source** | UCI Machine Learning Repository |
| **URL** | [https://archive.ics.uci.edu/dataset/942/rt-iot2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022) |
| **Total Records** | ~123,117 network flow records |
| **Feature Types** | Network-flow features (numerical and categorical) |
| **Target Variable** | `Attack_type` — the class label identifying normal traffic or specific attack type |
| **Task** | Multi-class classification |
| **Domain** | Network security / IoT |

The dataset contains a rich set of network flow-level features extracted from packet captures, including:

- **Flow-level statistics**: packet counts, byte counts, flow duration
- **Protocol information**: TCP/UDP flags, protocol types
- **Connection metadata**: source/destination ports, IP-related features
- **Statistical aggregates**: mean, standard deviation, min, max of inter-arrival times and packet sizes

Each record is labelled with an `Attack_type` value indicating the category of traffic, enabling supervised multi-class classification.

---

## Dataset Source

The RT-IoT2022 dataset is hosted on the **UCI Machine Learning Repository**, a widely recognised and trusted source of benchmark datasets for the machine learning research community.

- **Repository**: UCI Machine Learning Repository
- **Dataset Page**: [https://archive.ics.uci.edu/dataset/942/rt-iot2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022)
- **Year**: 2022
- **Format**: CSV

The dataset was captured in a controlled IoT network environment, incorporating realistic normal IoT traffic alongside a range of simulated network attacks. Full citation details are provided in the [Dataset Citation](#dataset-citation) section.

---

## Attack Categories

The RT-IoT2022 dataset includes **9 distinct attack types**, representing a diverse range of network intrusion techniques:

| # | Attack Type | Description |
|---|---|---|
| 1 | `DOS_SYN_Hping` | SYN flood denial-of-service attack using the Hping tool |
| 2 | `ARP_poisioning` | ARP cache poisoning / spoofing attack to intercept traffic |
| 3 | `NMAP_UDP_SCAN` | UDP port scanning using Nmap for service discovery |
| 4 | `NMAP_XMAS_TREE_SCAN` | Xmas tree scan (FIN+PSH+URG flags) using Nmap |
| 5 | `NMAP_OS_DETECTION` | Operating system fingerprinting using Nmap |
| 6 | `NMAP_TCP_scan` | TCP connect/SYN port scanning using Nmap |
| 7 | `DDOS_Slowloris` | Slowloris low-bandwidth DDoS attack targeting HTTP servers |
| 8 | `Metasploit_Brute_Force_SSH` | Brute-force SSH login attack using the Metasploit framework |
| 9 | `NMAP_FIN_SCAN` | FIN scan using Nmap to bypass firewall rules |

These attacks span multiple categories of the MITRE ATT&CK framework, including reconnaissance (scanning), credential access (brute force), and impact (denial of service).

---

## Normal Traffic Categories

The dataset includes **4 categories of legitimate IoT network traffic**:

| # | Normal Traffic Type | Description |
|---|---|---|
| 1 | `MQTT` | Message Queuing Telemetry Transport — lightweight IoT messaging protocol |
| 2 | `Thing_speak` | ThingSpeak — IoT analytics platform for sensor data collection |
| 3 | `Wipro_bulb_Dataset` | Traffic from Wipro smart bulb IoT devices |
| 4 | `Amazon-Alexa` | Traffic from Amazon Alexa voice-assistant IoT devices |

These categories represent realistic IoT device communication patterns, providing a meaningful baseline of normal network behaviour against which attacks can be distinguished.

---

## ML Methodology

This project employs **supervised multi-class classification** to categorise network traffic. The methodology follows established best practices in machine learning for cybersecurity:

1. **Data Loading & Exploration**: Load the RT-IoT2022 dataset, examine its structure, distributions, and characteristics.
2. **Data Splitting**: Split the data into training (80%) and testing (20%) sets using stratified sampling to preserve class proportions. `random_state=42` is used throughout for reproducibility.
3. **Preprocessing Pipeline**: Construct a `sklearn.pipeline.Pipeline` with a `ColumnTransformer` to handle categorical and numerical features appropriately, fitted exclusively on training data.
4. **Model Training**: Train multiple classifier architectures on the preprocessed training data, all configured with `class_weight="balanced"` to address class imbalance directly in the loss function.
5. **Model Comparison**: Compare all models on the held-out test set using macro F1-score as the primary metric.
6. **Hyperparameter Tuning**: Apply `RandomizedSearchCV` with `scoring="f1_macro"` and stratified cross-validation to tune the top-performing models.
7. **Final Evaluation**: Evaluate the best tuned model comprehensively with classification reports, confusion matrices, and feature importance analysis.
8. **Deployment**: Save the final model and preprocessing pipeline, and serve predictions through a Streamlit web application.

---

## Data Preprocessing

Proper preprocessing is critical for model performance and for preventing **data leakage**. The preprocessing pipeline is implemented using scikit-learn's `ColumnTransformer` and `Pipeline`:

### Categorical Features
- **Method**: `OneHotEncoder` (with `handle_unknown="ignore"` for robustness)
- **Rationale**: Converts categorical variables (e.g., protocol type, service) into binary indicator columns, enabling their use in numerical ML models without imposing ordinal relationships.

### Numerical Features
- **Method**: `StandardScaler` (zero mean, unit variance)
- **Rationale**: Normalises numerical features to a common scale, preventing features with larger magnitudes from dominating distance-based or gradient-based learning algorithms.

### Data Leakage Prevention
- The `ColumnTransformer` is fitted **only on the training data** using `pipeline.fit(X_train, y_train)`.
- The test data is transformed using the already-fitted pipeline via `pipeline.transform(X_test)`.
- This ensures that no information from the test set influences the preprocessing parameters (means, standard deviations, category mappings), preserving the integrity of the evaluation.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)  # Fit ONLY on training data
```

---

## Exploratory Data Analysis (EDA)

Comprehensive EDA is performed to understand the dataset before modelling:

1. **Class Distribution Analysis**: Visualise the frequency of each `Attack_type` category using bar charts to understand the class balance. This reveals significant imbalance, with some attack types having far fewer samples than others.

2. **Binary Distribution**: Aggregate classes into Normal (MQTT, Thing_speak, Wipro_bulb_Dataset, Amazon-Alexa) and Malicious (all attack types) for a binary perspective on the dataset composition.

3. **Correlation Analysis**: Compute and visualise the correlation matrix of numerical features using heatmaps to identify highly correlated feature pairs, which may indicate redundancy.

4. **Box Plots**: Generate box plots for key numerical features grouped by attack type to visualise distributional differences and identify potential discriminative features.

5. **Missing Value Analysis**: Check for missing values across all features. The RT-IoT2022 dataset is generally clean, but any missing values are identified and handled appropriately.

6. **Feature Type Summary**: Catalogue features by data type (numerical, categorical) to inform preprocessing decisions.

EDA outputs are saved to `outputs/figures/` for inclusion in reports and documentation.

---

## Class Imbalance Handling

The RT-IoT2022 dataset exhibits **significant class imbalance** — some attack types have substantially more samples than others, and the ratio between normal and attack traffic is uneven. Class imbalance is a critical challenge in intrusion detection because it can cause models to be biased toward majority classes, achieving high accuracy while failing to detect rare but dangerous attacks.

### Strategy: `class_weight="balanced"`

This project uses **`class_weight="balanced"`** as the primary strategy for handling class imbalance. This parameter is passed directly to scikit-learn classifiers and adjusts the loss function to assign higher penalties for misclassifying minority-class samples.

The weight for each class is computed as:

$$w_c = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_{\text{samples in class } c}}$$

### Why `class_weight="balanced"` Over Oversampling (e.g., SMOTE)?

| Criterion | `class_weight="balanced"` | SMOTE / Oversampling |
|---|---|---|
| **Data leakage risk** | None — operates on the loss function, not the data | High if applied before train/test split |
| **Implementation complexity** | Single parameter in model constructor | Requires additional library (imbalanced-learn), careful pipeline integration |
| **Compatibility** | Works with all scikit-learn classifiers | Not all models support synthetic samples well |
| **Overfitting risk** | Minimal | Synthetic samples can cause overfitting, especially in high-dimensional spaces |
| **Reproducibility** | Deterministic | Stochastic; results vary with random seed |

**Decision**: `class_weight="balanced"` was chosen because it is clean, introduces no risk of data leakage, integrates seamlessly into scikit-learn pipelines, and is effective across all model types used in this project. Blind oversampling without careful cross-validation was avoided to maintain methodological rigour.

---

## Models Used

Five supervised classification algorithms were trained and compared:

| # | Model | Description | Key Characteristics |
|---|---|---|---|
| 1 | **Logistic Regression** | Linear model using softmax for multi-class | Fast, interpretable, serves as a linear baseline |
| 2 | **Decision Tree** | Single tree with recursive binary splits | Highly interpretable, prone to overfitting |
| 3 | **Random Forest** | Ensemble of bagged decision trees | Robust, handles non-linearity, built-in feature importance |
| 4 | **HistGradientBoosting** | Histogram-based gradient boosting (similar to LightGBM) | Fast training on large datasets, native missing value support |
| 5 | **Extra Trees** | Extremely randomised trees ensemble | Similar to Random Forest but with more randomisation, often better generalisation |

All models were configured with `class_weight="balanced"` (or equivalent) and `random_state=42` for reproducibility.

---

## Hyperparameter Tuning

After initial model comparison, the **top 2 performing models** (based on macro F1-score on the test set) are selected for hyperparameter tuning using `RandomizedSearchCV`:

```python
from sklearn.model_selection import RandomizedSearchCV

tuner = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=5),
    random_state=42,
    n_jobs=-1,
    verbose=1
)
tuner.fit(X_train, y_train)
```

### Key Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `scoring` | `f1_macro` | Treats all classes equally regardless of size; appropriate for imbalanced data |
| `cv` | `StratifiedKFold(n_splits=5)` | Preserves class proportions in each fold |
| `n_iter` | 20 | Balances search thoroughness with computational cost |
| `random_state` | 42 | Ensures reproducibility |

### Typical Hyperparameter Search Spaces

- **Random Forest / Extra Trees**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **HistGradientBoosting**: `max_iter`, `max_depth`, `learning_rate`, `min_samples_leaf`, `max_leaf_nodes`

---

## Evaluation Metrics

Multiple evaluation metrics are computed to provide a comprehensive assessment of model performance:

| Metric | Description | Use Case |
|---|---|---|
| **Accuracy** | Proportion of correctly classified samples | Overall performance (but misleading for imbalanced data) |
| **Precision** | TP / (TP + FP) — of predicted positives, how many are correct | Important when false positives are costly |
| **Recall** | TP / (TP + FN) — of actual positives, how many are detected | Critical for IDS — missing an attack is dangerous |
| **F1-Score (Weighted)** | Harmonic mean of precision and recall, weighted by class support | Accounts for class size in overall score |
| **F1-Score (Macro)** | Unweighted mean of per-class F1-scores | Treats all classes equally — primary metric |
| **Classification Report** | Per-class precision, recall, F1-score, and support | Detailed breakdown of performance per attack type |
| **Confusion Matrix** | Matrix of actual vs. predicted labels | Visualises misclassification patterns |

### Why Accuracy Alone Is Misleading

In imbalanced datasets, a model can achieve high accuracy by simply predicting the majority class for every sample. For example, if 90% of traffic is normal, a model that always predicts "normal" achieves 90% accuracy — while detecting zero attacks. This is clearly unacceptable for an IDS.

**Macro F1-score** is used as the primary evaluation metric because it computes the F1-score independently for each class and then takes the unweighted average. This means that performance on rare attack types contributes equally to the overall score, ensuring that the model must perform well across all categories — including minority classes — to achieve a high macro F1-score.

---

## Final Model Selection

The final model is selected based on the **highest macro F1-score** achieved after hyperparameter tuning on the held-out test set.

The selection process is:

1. Train all 5 base models with default hyperparameters and `class_weight="balanced"`.
2. Evaluate each on the test set using macro F1-score.
3. Select the top 2 models for hyperparameter tuning.
4. Perform `RandomizedSearchCV` on each.
5. Compare tuned models on the test set.
6. Select the model with the highest macro F1-score as the final model.

The final model (along with its fitted preprocessing pipeline) is serialised using `joblib` and saved to `models/` for use in the prediction system and Streamlit application.

---

## Results

The models were trained and evaluated on the official **RT-IoT2022** dataset (123,117 total network flow records, 80/20 stratified train/test split with `random_state=42`, test set size = 24,624 records).

### Model Comparison Table

Evaluated on the held-out test split (24,624 samples):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 (Weighted) | F1 (Macro) | Training Time |
|---|---|---|---|---|---|---|
| **Decision Tree** | **0.9981** | **0.9981** | **0.9981** | **0.9981** | **0.9679** | 1.39s |
| **Random Forest** | 0.9984 | 0.9984 | 0.9984 | 0.9984 | 0.9604 | 3.10s |
| **Extra Trees** | 0.9985 | 0.9985 | 0.9985 | 0.9985 | 0.9599 | 4.51s |
| **HistGradientBoosting** | 0.9959 | 0.9963 | 0.9959 | 0.9960 | 0.8915 | 2.36s |
| **Logistic Regression** | 0.9884 | 0.9917 | 0.9884 | 0.9896 | 0.8237 | 27.98s |

*Generated at: `outputs/metrics/model_comparison.csv`*

### Final Selected Model Performance (Multi-Class)

The **Decision Tree** classifier with balanced class weighting was selected based on the highest **Macro F1-score (0.9679)**:

| Attack Category / Traffic Type | Precision | Recall | F1-Score | Support (Test Set) |
|---|---|---|---|---|
| **ARP_poisioning** | 0.99 | 0.99 | 0.99 | 1,550 |
| **DDOS_Slowloris** | 0.99 | 1.00 | 1.00 | 107 |
| **DOS_SYN_Hping** | 1.00 | 1.00 | 1.00 | 18,932 |
| **MQTT_Publish** *(Normal)* | 1.00 | 1.00 | 1.00 | 829 |
| **Metasploit_Brute_Force_SSH** | 0.78 | 1.00 | 0.88 | 7 |
| **NMAP_FIN_SCAN** | 0.83 | 0.83 | 0.83 | 6 |
| **NMAP_OS_DETECTION** | 1.00 | 1.00 | 1.00 | 400 |
| **NMAP_TCP_scan** | 1.00 | 1.00 | 1.00 | 200 |
| **NMAP_UDP_SCAN** | 0.99 | 0.99 | 0.99 | 518 |
| **NMAP_XMAS_TREE_SCAN** | 1.00 | 1.00 | 1.00 | 402 |
| **Thing_Speak** *(Normal)* | 0.99 | 0.99 | 0.99 | 1,622 |
| **Wipro_bulb** *(Normal)* | 0.98 | 0.92 | 0.95 | 51 |
| **Accuracy** | | | **0.9981** | 24,624 |
| **Macro Average** | **0.96** | **0.98** | **0.97** | 24,624 |
| **Weighted Average** | **1.00** | **1.00** | **1.00** | 24,624 |

*Generated at: `outputs/metrics/final_classification_report.csv`*

### Secondary Binary Evaluation (Normal vs Malicious)

Traffic categories mapped: `MQTT_Publish`, `Thing_Speak`, `Wipro_bulb` → **Normal**; all others → **Malicious**.

| Metric | Score |
|---|---|
| **Binary Accuracy** | **99.87%** (0.9987) |
| **Binary Precision** | **99.93%** (0.9993) |
| **Binary Recall** | **99.93%** (0.9993) |
| **Binary F1-Score** | **99.93%** (0.9993) |

*Generated at: `outputs/metrics/binary_evaluation.csv`*

### Generated Figures and Visualizations

All visual artifacts have been generated and saved:
- `outputs/figures/class_distribution.png`: Per-class sample counts across all 12 classes
- `outputs/figures/binary_distribution.png`: Normal vs Malicious breakdown (89.8% Malicious / 10.2% Normal)
- `outputs/figures/class_imbalance.png`: Imbalance ratios relative to the majority class
- `outputs/figures/confusion_matrix.png`: Multi-class confusion matrix
- `outputs/figures/confusion_matrix_binary.png`: Binary confusion matrix
- `outputs/figures/feature_importance.png`: Top 20 most important features
- `outputs/figures/feature_importance_top10.png`: Top 10 most important features
- `outputs/figures/correlation_matrix.png`: Heatmap of flow feature correlations
- `outputs/figures/boxplots.png`: Feature distributions across attack types

---

## Feature Importance

Understanding which features contribute most to classification decisions is essential for:

- **Interpretability**: Explaining model decisions to security analysts and stakeholders.
- **Security Insights**: Identifying which network flow characteristics are most indicative of specific attack types.
- **Feature Engineering**: Informing future feature selection and engineering efforts.

Tree-based models provide built-in feature importance scores based on the total reduction in impurity (Gini importance).

The project extracts and visualises the **Top 10 and Top 20 most important features** from the final model:

### Top 10 Most Important Features Measured

| Rank | Feature | Importance | Security Interpretation |
|---|---|---|---|
| 1 | `id.resp_p` | 0.1547 | Responding/destination port (e.g., port 1883 for MQTT, port 22 for SSH, scanning ports) |
| 2 | `fwd_last_window_size` | 0.1206 | TCP window size in forward direction; distinct signatures for port scanning vs DoS |
| 3 | `fwd_URG_flag_count` | 0.0909 | Urgent flag count; frequently manipulated in specialised scan types (e.g., XMAS scans) |
| 4 | `flow_SYN_flag_count` | 0.0907 | SYN flag volume; primary signature for `DOS_SYN_Hping` floods |
| 5 | `service_mqtt` | 0.0907 | Protocol service indicator; separates legitimate IoT broker traffic from malicious flows |
| 6 | `bwd_pkts_per_sec` | 0.0905 | Backward packet transmission rate; reveals response characteristics under DoS/scans |
| 7 | `service_http` | 0.0754 | HTTP protocol indicator; differentiates Slowloris application-layer attacks |
| 8 | `service_-` | 0.0723 | Unrecognised or raw service flows; common during port scans and raw socket injections |
| 9 | `flow_iat.min` | 0.0697 | Minimum inter-arrival time between packets; high frequency characterizes flooding tools |
| 10 | `fwd_subflow_pkts` | 0.0335 | Subflow packet volume; characterizes bursty traffic patterns of automated attacks |

*Complete rankings saved to: `outputs/metrics/feature_importance.csv`*

---

## Prediction System

The prediction system provides an end-to-end pipeline for classifying new network traffic data:

```
Upload CSV → Validate → Preprocess → Predict → Output
```

### Pipeline Steps

1. **Upload CSV**: The user provides a CSV file containing network flow records (same feature schema as the training data, without the `Attack_type` column).

2. **Validate**: The system checks:
   - File format is valid CSV
   - Required columns are present
   - Data types are compatible
   - No structural inconsistencies

3. **Preprocess**: The uploaded data is transformed using the **same fitted preprocessing pipeline** from training (loaded from the saved model artifact). This ensures consistency between training and inference.

4. **Predict**: The trained model generates:
   - **`Attack_Type`**: The predicted multi-class label (e.g., `MQTT`, `DOS_SYN_Hping`)
   - **`Binary_Label`**: `Normal` or `Malicious` based on whether the predicted class is a normal traffic type or an attack type
   - **`Confidence`**: The probability of the predicted class (from `predict_proba`)

5. **Output**: Results are displayed in a table and available for download as a CSV file.

---

## Streamlit Application

The project includes an interactive **Streamlit web application** (`app.py`) that provides a user-friendly interface for the prediction system.

### Features

- **File Upload**: Drag-and-drop or browse to upload a CSV file of network traffic data.
- **Automatic Prediction**: Upon upload, the data is validated, preprocessed, and classified automatically.
- **Results Table**: Displays predictions with Attack_Type, Binary_Label (Normal/Malicious), and Confidence for each flow.
- **Download Results**: Export prediction results as a CSV file.
- **Model Information**: Displays details about the loaded model and its performance metrics.

### Running the Application

```bash
streamlit run app.py
```

The application launches in your default web browser at `http://localhost:8501`.

---

## Project Structure

```
AIML Project/
├── data/
│   └── README.md
├── docs/
│   └── methodology.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── models/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── app.py
├── requirements.txt
├── README.md
├── AGENTS.md
└── .gitignore
```

| Directory / File | Purpose |
|---|---|
| `data/` | Dataset storage (CSV files); not committed to version control |
| `docs/` | Project documentation and methodology notes |
| `notebooks/` | Jupyter notebooks for interactive exploration and experimentation |
| `src/` | Python source modules — the core ML pipeline |
| `src/config.py` | Centralised configuration (paths, parameters, constants) |
| `src/data_loader.py` | Data loading and initial validation |
| `src/preprocessing.py` | Feature preprocessing (ColumnTransformer, Pipeline) |
| `src/eda.py` | Exploratory data analysis and visualisation |
| `src/train.py` | Model training, comparison, and hyperparameter tuning |
| `src/evaluate.py` | Model evaluation on test data |
| `src/predict.py` | Prediction on new/unseen data |
| `src/utils.py` | Utility functions (logging, file I/O, helpers) |
| `models/` | Saved model artifacts (joblib files) |
| `outputs/figures/` | EDA and evaluation plots (PNG) |
| `outputs/metrics/` | Classification reports and comparison tables (CSV) |
| `outputs/predictions/` | Prediction output files |
| `app.py` | Streamlit web application |
| `requirements.txt` | Python package dependencies |
| `AGENTS.md` | Agent configuration and project rules |
| `.gitignore` | Git ignore rules |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "AIML Project"

# (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

The RT-IoT2022 dataset must be downloaded separately from the UCI Machine Learning Repository and placed in the `data/` directory.

1. Visit [https://archive.ics.uci.edu/dataset/942/rt-iot2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022)
2. Download the dataset archive.
3. Extract the CSV file and place it as:

```
data/RT_IoT2022.csv
```

> **Note**: The dataset file is not included in the repository due to its size. It must be downloaded independently.

---

## How to Train

Run the complete training pipeline:

```bash
python src/train.py
```

This will:
1. Load and explore the dataset
2. Perform EDA and save visualisations to `outputs/figures/`
3. Split data into training and testing sets (80/20, stratified)
4. Preprocess features using the fitted ColumnTransformer pipeline
5. Train all 5 models with `class_weight="balanced"`
6. Compare models and select the top 2 for hyperparameter tuning
7. Perform `RandomizedSearchCV` on the top models
8. Evaluate the best model and save results to `outputs/metrics/`
9. Save the final model to `models/`

---

## How to Evaluate

Evaluate the saved model on the test set:

```bash
python src/evaluate.py
```

This loads the trained model from `models/`, evaluates it on the held-out test set, and produces:
- Classification report (console output and CSV)
- Confusion matrix heatmap
- Per-class performance breakdown

---

## How to Run the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Upload a CSV file of network traffic data to receive predictions.

---

## How to Predict from CLI

Run predictions on a new CSV file from the command line:

```bash
python src/predict.py data/new_traffic.csv
```

The output CSV (with `Attack_Type`, `Binary_Label`, and `Confidence` columns appended) is saved to `outputs/predictions/`.

---

## Limitations

It is important to understand the scope and limitations of this project:

- **Dataset-based ML IDS, not a production IDS**: This system is a research prototype trained on a static dataset. It is not a fully operational, production-grade intrusion detection system.

- **Does NOT capture live network packets**: The system classifies pre-extracted network flow records provided as CSV files. It does not perform packet capture, deep packet inspection, or real-time traffic monitoring.

- **"RT" in RT-IoT2022 refers to the dataset name, not real-time capability**: The dataset name should not be interpreted as indicating that this system operates in real time. The "RT" prefix is part of the dataset's naming convention.

- **Real deployment requires integration with network monitoring systems**: To use this model in a live environment, it would need to be integrated with network monitoring tools (e.g., Zeek, Suricata, Wireshark) that extract flow-level features from raw packets in real time.

- **Trained on specific attack types; may not generalise to novel attacks**: The model is trained to recognise the 9 attack types present in the RT-IoT2022 dataset. It may fail to detect attack types not represented in the training data (zero-day attacks, APTs, etc.).

- **Performance depends on feature quality**: The model's effectiveness is contingent on the quality and relevance of the network flow features. In different network environments or with different feature extraction methods, performance may vary.

---

## Future Work

Several directions could extend and improve this project:

1. **Real-Time Packet Ingestion**: Integrate with packet capture tools (e.g., Zeek, tshark) to extract flow features from live network traffic and classify them in real time.

2. **Online Learning**: Implement incremental / online learning algorithms that can update the model as new traffic patterns emerge without full retraining.

3. **Concept Drift Detection**: Add monitoring for concept drift — detecting when the statistical properties of network traffic change over time, signalling that the model may need retraining.

4. **Additional Datasets**: Train and evaluate on complementary datasets (e.g., CICIDS2017, NSL-KDD, UNSW-NB15) to improve generalisation across different network environments.

5. **Explainable AI (XAI)**: Incorporate SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) for per-prediction explanations, helping analysts understand why a specific flow was flagged.

6. **Model Monitoring and MLOps**: Implement model performance monitoring in production, including automated alerts for performance degradation, A/B testing of model versions, and automated retraining pipelines.

7. **Adversarial Machine Learning**: Investigate the model's robustness against adversarial evasion attacks, where an attacker crafts traffic to evade detection, and implement adversarial training or detection mechanisms.

8. **Edge Deployment**: Optimise and deploy the model on edge devices (e.g., Raspberry Pi, NVIDIA Jetson) for on-device IoT network monitoring with minimal latency.

9. **SIEM Integration**: Integrate the prediction system with Security Information and Event Management (SIEM) platforms (e.g., Splunk, Elastic SIEM) for centralised alerting and incident response.

10. **Automated Alert System**: Build an alerting mechanism that notifies security teams via email, Slack, or webhook when malicious traffic is detected, with configurable severity thresholds.

---

## Ethical and Security Considerations

- **Responsible Use**: This project is developed for **defensive security purposes only** — to detect and classify network intrusions, protect IoT infrastructure, and advance cybersecurity research. It must not be used for offensive purposes, unauthorised surveillance, or any activity that violates laws or ethical standards.

- **Controlled Environment**: The RT-IoT2022 dataset was captured in a **controlled laboratory environment** specifically set up for security research. The attacks were simulated in an isolated network with no impact on real users or systems.

- **Privacy Considerations**: While the dataset contains network flow metadata (packet counts, byte counts, timing), it does not include payload content or personally identifiable information (PII). However, in a real deployment, care must be taken to comply with data protection regulations (e.g., GDPR, DPA 2018) when processing network traffic data.

- **Bias and Fairness**: The model's performance is inherently bounded by the dataset's coverage. It may exhibit biased behaviour toward traffic patterns not represented in the training data. Continuous evaluation and dataset diversification are necessary for production use.

- **Transparency**: The use of interpretable models and feature importance analysis supports transparency in automated security decision-making, aligning with principles of explainable AI.

---

## Dataset Citation

If you use this project or the RT-IoT2022 dataset in your research, please cite:

> **RT-IoT2022 Dataset**
> UCI Machine Learning Repository
> [https://archive.ics.uci.edu/dataset/942/rt-iot2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022)
>
> B. S. Sharmila and Rohini Nagapadma. (2023). *RT-IoT2022*. UCI Machine Learning Repository. https://doi.org/10.24432/C5P329

```bibtex
@misc{rt_iot2022,
  author       = {Sharmila, B. S. and Nagapadma, Rohini},
  title        = {{RT-IoT2022}},
  year         = {2023},
  publisher    = {UCI Machine Learning Repository},
  url          = {https://archive.ics.uci.edu/dataset/942/rt-iot2022},
  doi          = {10.24432/C5P329}
}
```

---

## Viva Questions and Answers

This section presents 20 questions and answers covering the key concepts, design decisions, and technical details of this project. These are intended to prepare for academic viva / defence discussions.

---

### Q1: What is an Intrusion Detection System (IDS)?

**A:** An Intrusion Detection System (IDS) is a security mechanism that monitors network traffic or system activities for malicious behaviour or policy violations. There are two main types: **Network-based IDS (NIDS)**, which monitors traffic flowing across the network, and **Host-based IDS (HIDS)**, which monitors activity on individual devices. IDS can be further categorised as **signature-based** (matching known attack patterns) or **anomaly-based** (detecting deviations from a learned baseline of normal behaviour). This project implements a network-based, anomaly/signature hybrid approach using supervised machine learning, where the model learns patterns from labelled examples of both normal and attack traffic.

---

### Q2: Why is machine learning useful for IDS?

**A:** Machine learning is valuable for IDS because it can automatically learn complex, non-linear patterns from data that would be difficult or impossible to encode as manual rules. Unlike signature-based systems that require constant updates with known attack patterns, ML models can generalise from training examples to detect variants of known attacks and, in some cases, previously unseen attack types. ML models can process high-dimensional feature spaces (dozens of network flow attributes simultaneously), scale to large volumes of traffic, and adapt to evolving network conditions. They also reduce the reliance on expert-crafted rules, which are time-consuming to develop and maintain.

---

### Q3: Why was the RT-IoT2022 dataset selected?

**A:** The RT-IoT2022 dataset was selected for several reasons: (1) It is a **modern dataset** (2022), reflecting current IoT protocols and attack techniques, unlike older datasets such as KDD'99 or NSL-KDD that contain outdated traffic patterns. (2) It is **IoT-specific**, capturing realistic traffic from actual IoT devices (MQTT sensors, smart bulbs, Alexa), making it directly relevant to IoT security research. (3) It includes a **diverse set of attack types** (9 categories spanning DoS, scanning, ARP poisoning, and brute force). (4) It is hosted on the **UCI Machine Learning Repository**, a reputable and widely cited source. (5) It has a manageable size (~123,117 records) suitable for academic research without requiring distributed computing infrastructure.

---

### Q4: What is the target variable and what does it represent?

**A:** The target variable is **`Attack_type`**. It is a categorical variable that labels each network flow record with its traffic category. This includes 9 attack types (e.g., `DOS_SYN_Hping`, `ARP_poisioning`, `NMAP_UDP_SCAN`) and 4 normal traffic types (e.g., `MQTT`, `Amazon-Alexa`). In multi-class classification, the model predicts one of these 13 categories for each flow. A binary interpretation is also derived by mapping attack categories to "Malicious" and normal categories to "Normal", enabling a simple attack/no-attack decision.

---

### Q5: What is multi-class classification?

**A:** Multi-class classification is a supervised learning task where the model predicts one of **three or more** discrete class labels for each input sample. In this project, the model classifies each network flow into one of 13 categories (9 attack types + 4 normal types). This is in contrast to binary classification (two classes) or regression (continuous output). Multi-class classification requires algorithms capable of handling multiple decision boundaries, such as one-vs-rest logistic regression, decision trees, or ensemble methods like Random Forest that naturally support multiple classes.

---

### Q6: What is binary classification and how does this project use it?

**A:** Binary classification is a supervised learning task with exactly **two** class labels (e.g., positive/negative, yes/no). While this project's primary task is multi-class classification (13 classes), a **binary interpretation** is derived by mapping the predicted multi-class label to either "Normal" (if the prediction is one of the 4 normal traffic types: MQTT, Thing_speak, Wipro_bulb_Dataset, Amazon-Alexa) or "Malicious" (if the prediction is any of the 9 attack types). This provides security analysts with a quick, actionable assessment — is this flow an attack or not? — while still retaining the granularity of the specific attack type.

---

### Q7: Why is the dataset imbalanced?

**A:** The dataset is imbalanced because the number of samples varies significantly across classes. This reflects the **real-world distribution** of network traffic: normal traffic typically vastly outnumbers any single attack type, and different attack campaigns generate different volumes of malicious flows depending on their duration and intensity. Some attack types (e.g., scans) may produce many short flows, while others (e.g., brute force) may produce fewer but longer sessions. This imbalance is inherent to cybersecurity datasets and must be addressed during modelling to prevent the classifier from being biased toward majority classes.

---

### Q8: Why can accuracy be misleading for imbalanced datasets?

**A:** Accuracy measures the proportion of correct predictions overall: (TP + TN) / Total. In imbalanced datasets, the majority class dominates this metric. For example, if 90% of samples are normal traffic, a model that **always predicts "normal"** (never detecting any attack) achieves 90% accuracy — a seemingly impressive number that is completely useless for intrusion detection. Accuracy fails to reveal that the model has 0% recall on attack classes. This is why metrics like **precision, recall, F1-score (especially macro F1)**, and **confusion matrices** are essential — they evaluate performance on each class independently, ensuring that minority classes (rare attacks) are not ignored.

---

### Q9: What are precision, recall, and F1-score?

**A:**
- **Precision** = TP / (TP + FP): Of all samples the model predicted as a particular class, what fraction actually belongs to that class? High precision means few false positives. For IDS, high precision for an attack class means few false alarms.
- **Recall** = TP / (TP + FN): Of all samples that actually belong to a class, what fraction did the model correctly identify? High recall means few missed detections. For IDS, high recall for attack classes is critical — missing an attack can be catastrophic.
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall): The harmonic mean of precision and recall, providing a single balanced metric. It is high only when both precision and recall are high, making it especially useful when there is an uneven class distribution.

---

### Q10: Why use macro F1-score as the primary metric?

**A:** Macro F1-score computes the F1-score independently for each class and then takes the **unweighted average**. This means that every class — whether it has 10,000 samples or 100 — contributes equally to the final score. For intrusion detection, this is crucial because rare attack types are just as important (or more important) to detect as common ones. A model that performs well on majority classes but poorly on minority attack types would receive a low macro F1-score, correctly reflecting its inadequacy. In contrast, weighted F1 would mask poor performance on rare classes by giving them less weight proportional to their support.

---

### Q11: Why use Random Forest / tree-based models for IDS?

**A:** Tree-based models are well-suited for network intrusion detection for several reasons: (1) They handle **mixed feature types** (numerical and categorical) naturally. (2) They capture **non-linear relationships** and complex feature interactions without manual feature engineering. (3) They are **robust to irrelevant features** and can identify the most discriminative attributes. (4) Ensemble methods (Random Forest, Extra Trees) reduce overfitting through averaging/voting across many diverse trees. (5) They provide **built-in feature importance** scores, enabling interpretability — security analysts can understand which network flow features drive detection. (6) They are **computationally efficient** for training and inference, making them practical for real-world deployment.

---

### Q12: What is feature importance and why does it matter?

**A:** Feature importance quantifies the **contribution of each input feature** to the model's predictions. In tree-based models, this is typically measured by the total reduction in impurity (Gini importance) averaged across all trees and splits involving that feature, or by permutation importance (measuring how much performance degrades when a feature's values are shuffled). Feature importance matters for IDS because: (1) It provides **interpretability** — analysts can understand what the model is "looking at" when making decisions. (2) It offers **security insights** — knowing that packet count or specific TCP flags are highly discriminative informs network monitoring strategies. (3) It supports **feature selection** — removing uninformative features can reduce model complexity and improve generalisation.

---

### Q13: What is data leakage and how did you prevent it?

**A:** Data leakage occurs when information from outside the training set (particularly from the test set or the target variable) inadvertently influences the model during training, leading to **overly optimistic performance estimates** that do not generalise to real-world data. In this project, data leakage is prevented by: (1) **Splitting data before any preprocessing**: The train/test split is performed on the raw data before any transformations. (2) **Fitting the preprocessor only on training data**: The `StandardScaler` and `OneHotEncoder` within the `ColumnTransformer` are fitted exclusively on `X_train`, and then used to transform both `X_train` and `X_test`. This ensures that test data statistics (means, variances, category sets) do not influence the pipeline. (3) **Using scikit-learn Pipelines**: The `Pipeline` object encapsulates preprocessing and model training, ensuring the correct fit/transform sequence. (4) **Using `class_weight="balanced"` instead of SMOTE**: Oversampling applied before splitting would create synthetic copies of test samples in the training set — a direct form of data leakage.

---

### Q14: Why split data into training and testing sets?

**A:** Splitting data into training and testing sets simulates the real-world scenario where the model encounters **previously unseen data**. The training set is used to learn patterns, and the test set evaluates how well the model generalises to new, unseen examples. Without this split, the model would be evaluated on the same data it was trained on, providing an overly optimistic estimate of performance (especially for flexible models like decision trees that can memorise training data). In this project, an 80/20 stratified split is used, ensuring that class proportions are preserved in both sets, which is critical given the class imbalance.

---

### Q15: Why use cross-validation during hyperparameter tuning?

**A:** Cross-validation provides a more **robust and reliable estimate** of model performance than a single train/test split. In k-fold cross-validation, the training data is divided into k folds; the model is trained on k-1 folds and evaluated on the remaining fold, repeated k times. This: (1) Reduces variance in performance estimates — the result is averaged over k evaluations. (2) Uses all training data for both training and validation (each sample appears in the validation fold exactly once). (3) Helps detect overfitting — a model that performs well on training folds but poorly on validation folds is overfitting. **Stratified** cross-validation is used here to ensure each fold preserves the original class distribution, which is essential for imbalanced data.

---

### Q16: Why encode categorical features?

**A:** Most machine learning algorithms operate on numerical data. Categorical features (e.g., protocol type, service name) are represented as text strings that have no inherent numerical meaning. **One-hot encoding** converts each categorical feature into a set of binary columns — one per category — where a 1 indicates the presence of that category and 0 indicates absence. This avoids imposing a false ordinal relationship (e.g., treating protocol A < protocol B < protocol C numerically), which methods like label encoding would introduce. One-hot encoding preserves the categorical nature of the data while making it amenable to numerical computation.

---

### Q17: Why scale numerical features?

**A:** Numerical features in network flow data can have vastly different scales — for example, packet counts may range from 1 to millions, while inter-arrival times may be measured in microseconds. **StandardScaler** normalises each feature to have zero mean and unit variance (z-score normalisation). This is important because: (1) **Distance-based algorithms** (e.g., logistic regression with regularisation) are sensitive to feature magnitudes — unscaled features with large values would dominate the loss function. (2) **Gradient-based optimisation** converges faster when features are on similar scales. (3) While tree-based models are technically scale-invariant (they split on thresholds), scaling ensures consistency across the pipeline and allows fair comparison with linear models.

---

### Q18: Why should SMOTE / oversampling only be applied to training data?

**A:** SMOTE (Synthetic Minority Over-sampling Technique) generates **synthetic samples** by interpolating between existing minority-class examples. If SMOTE is applied to the entire dataset before the train/test split, synthetic samples created from test-set instances may end up in the training set. This is a form of **data leakage** because the model effectively "sees" information derived from test data during training. The result is inflated performance metrics that do not reflect real-world generalisation. If oversampling is used, it must be applied **only within the training set** (or within each cross-validation fold), after the split. In this project, `class_weight="balanced"` is preferred precisely because it avoids this risk entirely — it adjusts the loss function without modifying the data.

---

### Q19: What are the limitations of a dataset-based IDS?

**A:** A dataset-based IDS has several inherent limitations: (1) **Static training data**: The model is trained on a fixed dataset and cannot adapt to new attack patterns without retraining. (2) **Feature dependency**: Performance is tied to the specific set of features extracted; different feature extraction tools or network configurations may produce incompatible features. (3) **No real-time capability**: The system classifies pre-extracted flow records, not live packets. Real-time operation requires integration with network monitoring infrastructure. (4) **Dataset bias**: The model reflects the biases and limitations of its training dataset — attack types, traffic volumes, and network conditions in the dataset may not represent all production environments. (5) **No deep packet inspection**: Flow-level features summarise connections statistically; they do not inspect payload content, potentially missing application-layer attacks. (6) **Concept drift**: Network traffic patterns evolve over time, and a static model may degrade in performance without periodic retraining.

---

### Q20: How would you deploy this in a real network environment?

**A:** Deploying this IDS in a real network environment would involve several steps: (1) **Integrate with a network tap or mirror port** to capture live network traffic without disrupting normal operations. (2) **Use a flow extraction tool** (e.g., Zeek/Bro, CICFlowMeter, nProbe) to convert raw packets into flow-level feature records matching the training data schema. (3) **Deploy the trained model** as a microservice or API (e.g., using Flask, FastAPI, or a containerised service) that accepts flow records and returns predictions. (4) **Connect to a SIEM** (e.g., Splunk, Elastic Security) for centralised alerting, log correlation, and incident response workflows. (5) **Implement automated alerting** via email, Slack, or PagerDuty when malicious traffic is detected above configurable confidence thresholds. (6) **Set up model monitoring** to track prediction distributions, latency, and accuracy over time, triggering retraining when concept drift is detected. (7) **Ensure compliance** with organisational security policies, data protection regulations (GDPR), and network access controls.

---

## Acknowledgements

- **Dataset**: B. S. Sharmila and Rohini Nagapadma — RT-IoT2022, UCI Machine Learning Repository
- **Tools**: Python, scikit-learn, pandas, matplotlib, seaborn, Streamlit, joblib
- **Platform**: UCI Machine Learning Repository

---

*This project was developed as part of an MSc Cybersecurity programme. For questions or contributions, please open an issue or submit a pull request.*
