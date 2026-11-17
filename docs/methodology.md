# Methodology: Machine Learning-Based Network Intrusion Detection

## Abstract

This document presents the complete methodology employed in the development of a machine learning-based Network Intrusion Detection System (NIDS) for classifying malicious network traffic. The system is designed to address the growing security challenges in Internet of Things (IoT) environments by leveraging supervised learning techniques on the RT-IoT2022 dataset. The methodology encompasses the full machine learning lifecycle—from problem formulation and data acquisition through to model deployment and real-time prediction.

---

## 1. Problem Definition

### 1.1 Context

The proliferation of IoT devices across critical infrastructure, smart homes, and industrial systems has dramatically expanded the attack surface available to adversaries. Network Intrusion Detection Systems (NIDS) serve as a vital line of defence, monitoring network traffic to identify potentially malicious activity. However, traditional signature-based detection systems suffer from fundamental limitations:

- **Inability to detect novel attacks**: Signature-based systems rely on pre-defined patterns of known threats, rendering them ineffective against zero-day exploits and previously unseen attack vectors.
- **High maintenance overhead**: Security analysts must continuously update signature databases to remain effective against an evolving threat landscape.
- **Poor adaptability to IoT traffic**: IoT environments generate diverse and heterogeneous traffic patterns that are difficult to capture with rigid rule sets.

### 1.2 Proposed Solution

This project addresses these limitations by developing a **supervised machine learning classifier** capable of distinguishing between normal IoT network traffic and multiple categories of cyberattacks. Unlike signature-based approaches, ML-based classifiers learn statistical patterns from labelled data, enabling them to generalise to unseen traffic instances and provide a more adaptive detection capability.

### 1.3 Research Objective

The primary objective is to build, evaluate, and deploy an ML pipeline that achieves high classification performance across all attack types—including minority classes—while maintaining reproducibility and preventing data leakage throughout the experimental process.

---

## 2. Dataset Collection

### 2.1 Data Source

The **RT-IoT2022** dataset, sourced from the UCI Machine Learning Repository, serves as the foundation for this study. This dataset was specifically constructed to reflect realistic IoT network traffic conditions, incorporating both benign and malicious traffic captured in a controlled environment.

### 2.2 Dataset Characteristics

| Property | Value |
|---|---|
| **Total Records** | ~123,117 network traffic instances |
| **Traffic Composition** | Mix of normal IoT device communications and attack scenarios |
| **Attack Types** | Multiple categories including DoS, DDoS, brute force, and other attack vectors |
| **Features** | Network flow-level attributes (packet counts, byte counts, flow durations, protocol flags, etc.) |
| **Labels** | Fully labelled with attack type annotations |

### 2.3 Suitability Justification

The RT-IoT2022 dataset was selected for the following reasons:

1. **Realistic IoT traffic**: The dataset captures genuine IoT communication patterns rather than synthetic or simulated traffic, enhancing the ecological validity of the trained models.
2. **Multi-class labelling**: The availability of fine-grained attack type labels permits multi-class classification rather than simple binary detection, offering greater operational utility.
3. **Sufficient volume**: With over 123,000 records, the dataset provides adequate training samples to learn discriminative patterns while reserving sufficient data for evaluation.
4. **Contemporary relevance**: As a 2022 dataset, it reflects modern attack methodologies and IoT communication protocols.

---

## 3. Data Understanding

### 3.1 Initial Inspection

Prior to any transformation, a thorough inspection of the raw data was conducted to understand its structure and identify potential issues. This phase included:

- **Shape analysis**: Examining the number of records (rows) and features (columns) to establish the dimensionality of the classification problem.
- **Data type assessment**: Identifying the type of each feature—numerical (integer or float) or categorical (string/object)—to inform subsequent preprocessing decisions.
- **Statistical summaries**: Computing descriptive statistics (mean, standard deviation, min, max, quartiles) for numerical features to detect anomalies, outliers, or unexpected value ranges.

### 3.2 Feature Categorisation

Features were programmatically categorised into two groups:

- **Numerical features**: Continuous or discrete variables representing quantities such as packet counts, byte volumes, flow durations, and protocol-specific metrics.
- **Categorical features**: Nominal variables representing discrete categories such as protocol types, service identifiers, or flag states.

### 3.3 Target Variable

The target variable for classification is **`Attack_type`**, which encodes the category of network traffic (normal or specific attack type) for each record. This multi-class target forms the basis of the supervised learning task.

### 3.4 Column Exclusion

Any **ID-like columns** (e.g., unique identifiers, timestamps that serve as indices) were programmatically identified and excluded. Such columns carry no predictive information and, if retained, could lead to spurious correlations or overfitting.

---

## 4. Data Preprocessing

### 4.1 Rationale

Raw network traffic data frequently contains artefacts that can impair model training. Preprocessing transforms the data into a clean, consistent representation suitable for machine learning algorithms, and is an essential prerequisite for reliable model performance.

### 4.2 Missing and Infinite Values

- **Missing values (NaN)**: Records or features with missing values were identified and handled appropriately (e.g., imputation or removal) to prevent errors during model fitting.
- **Infinite values (±∞)**: Network flow computations (such as rates and ratios) can produce infinite values when denominators are zero. These were detected and replaced with finite representations (typically NaN, then handled alongside other missing values).

### 4.3 ID Column Removal

Columns identified as identifiers during the data understanding phase were removed from the feature set to prevent the model from memorising instance-specific artefacts rather than learning generalisable traffic patterns.

### 4.4 Feature-Target Separation

The dataset was separated into:

- **X (feature matrix)**: All predictor variables used as input to the models.
- **y (target vector)**: The `Attack_type` label for each instance.

This separation is a standard prerequisite for supervised learning, enabling independent manipulation of features and labels during training and evaluation.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Purpose

Exploratory Data Analysis provides critical insight into the structure, distribution, and relationships within the data. The findings from EDA directly inform modelling decisions—such as the choice of algorithms, the handling of class imbalance, and the selection of evaluation metrics.

### 5.2 Analyses Conducted

#### 5.2.1 Class Distribution Analysis

The frequency of each attack type was computed and visualised to understand the composition of the dataset. This analysis revealed the proportion of normal traffic relative to each attack category.

#### 5.2.2 Normal vs. Malicious Distribution

A binary aggregation was performed, grouping all attack types as *Malicious* and legitimate IoT traffic as *Normal*. This provided a high-level view of the overall balance between benign and adversarial traffic.

#### 5.2.3 Numerical Feature Distributions

Histograms and density plots were generated for numerical features to assess their distributional characteristics—including skewness, multimodality, and the presence of outliers. These distributions informed decisions regarding feature scaling.

#### 5.2.4 Correlation Analysis

Pairwise correlations among numerical features were computed to identify highly correlated (redundant) feature pairs. Strong inter-feature correlations may indicate multicollinearity, which can affect the stability of certain models (e.g., Logistic Regression).

#### 5.2.5 Class Imbalance Assessment

A quantitative assessment of class imbalance was conducted, measuring the ratio between the most frequent and least frequent classes. Significant imbalance—where certain attack types are substantially underrepresented—directly impacts model training and necessitates mitigation strategies.

---

## 6. Train/Test Split

### 6.1 Split Configuration

The dataset was partitioned into training and test subsets using the following configuration:

| Parameter | Value | Justification |
|---|---|---|
| **Train proportion** | 80% | Provides sufficient data for model learning |
| **Test proportion** | 20% | Reserves adequate unseen data for unbiased evaluation |
| **Stratification** | By `Attack_type` | Preserves class distribution in both splits |
| **Random state** | 42 | Ensures reproducibility across experiments |

### 6.2 Why Stratification Is Crucial

In datasets with imbalanced class distributions, a naïve random split may result in minority classes being entirely absent from either the training or test set. **Stratified splitting** ensures that the proportional representation of each attack type is maintained in both partitions. This is essential because:

- The model must observe examples of all attack types during training.
- The test set must contain instances of all classes to provide a meaningful evaluation across the full classification taxonomy.

### 6.3 Why a Fixed Random State Ensures Reproducibility

Setting `random_state=42` guarantees that the same partition is generated on every execution. This is a fundamental requirement for scientific reproducibility, enabling:

- Consistent comparison across different models trained on identical data.
- Verification and replication of results by independent researchers.
- Deterministic debugging and iterative improvement of the pipeline.

---

## 7. Class Imbalance Handling

### 7.1 The Problem

Network traffic datasets typically exhibit significant class imbalance: normal traffic far outnumbers any individual attack type, and some attack categories may have very few recorded instances. If left unaddressed, this imbalance causes classifiers to develop a bias towards the majority class, achieving high overall accuracy while failing to detect rare—but potentially critical—attack types.

### 7.2 Strategy: Class Weight Balancing

The primary mitigation strategy employs the **`class_weight="balanced"`** parameter, which is natively supported by scikit-learn classifiers. This approach adjusts the loss function during training to assign higher penalties for misclassifying minority classes, computed as:

$$w_c = \frac{n_{\text{total}}}{n_{\text{classes}} \times n_c}$$

where $w_c$ is the weight for class $c$, $n_{\text{total}}$ is the total number of samples, $n_{\text{classes}}$ is the number of distinct classes, and $n_c$ is the number of samples in class $c$.

### 7.3 Why Not SMOTE as the Primary Strategy

While Synthetic Minority Over-sampling Technique (SMOTE) is a widely used alternative, `class_weight="balanced"` was preferred for the following reasons:

- **Simplicity**: No additional data generation step is required, reducing pipeline complexity.
- **No leakage risk**: SMOTE generates synthetic samples that, if applied before the train/test split, can introduce information from test instances into the training set—a critical methodological error. Cost-sensitive learning via class weights operates solely on the loss function and carries no such risk.
- **Universal compatibility**: Class weighting is supported by all classifiers in this study, whereas SMOTE requires compatible model architectures and careful integration into cross-validation folds.

> **Important**: Any oversampling technique, if employed, must be applied **exclusively to the training data**. Applying oversampling before the train/test split constitutes data leakage and produces artificially inflated performance metrics that do not reflect real-world detection capability.

---

## 8. Feature Transformation

### 8.1 Transformation Architecture

Feature transformation was implemented using scikit-learn's **`ColumnTransformer`**, which applies distinct preprocessing pipelines to different feature subsets within a unified framework.

| Feature Type | Transformer | Rationale |
|---|---|---|
| **Numerical** | `StandardScaler` | Centres features to zero mean and unit variance, essential for distance-based and gradient-based algorithms (e.g., Logistic Regression) that are sensitive to feature magnitudes |
| **Categorical** | `OneHotEncoder` | Converts nominal categories into binary indicator variables, as ML algorithms require numerical input and cannot natively interpret string labels |

### 8.2 Pipeline Architecture and Data Leakage Prevention

The `ColumnTransformer` is embedded within a scikit-learn **`Pipeline`**, which chains preprocessing and classification into a single estimator object. This architectural decision is critical for preventing data leakage:

- **The scaler and encoder are fitted exclusively on the training data.** The learned parameters (means, standard deviations, category mappings) are then applied to transform both the training and test data.
- If the scaler were fitted on the entire dataset (including test data), the model would gain indirect access to information about the test distribution, leading to optimistically biased performance estimates.

### 8.3 Transformation Workflow

```
Training Phase:
  Training Data → fit_transform(X_train) → Scaled/Encoded Training Data → Model.fit()

Evaluation Phase:
  Test Data → transform(X_test) → Scaled/Encoded Test Data → Model.predict()
```

The `fit_transform` operation learns and applies the transformation on training data, while `transform` applies the previously learned transformation to test (or new) data without re-fitting.

---

## 9. Model Training

### 9.1 Model Selection Rationale

Five supervised learning algorithms were selected to provide a comprehensive evaluation across different model families, balancing interpretability, computational efficiency, and predictive power:

| # | Model | Family | Key Strengths |
|---|---|---|---|
| 1 | **Logistic Regression** | Linear | Serves as a linear baseline; provides probability estimates; well-understood theoretical properties |
| 2 | **Decision Tree** | Tree-based | Highly interpretable; captures non-linear decision boundaries; no feature scaling required |
| 3 | **Random Forest** | Ensemble (bagging) | Reduces variance through bootstrap aggregation; handles high-dimensional feature spaces; robust to overfitting |
| 4 | **HistGradientBoosting** | Ensemble (boosting) | State-of-the-art gradient boosting with native handling of missing values; histogram-based splitting for efficiency |
| 5 | **Extra Trees** | Ensemble (bagging) | Introduces additional randomisation in split selection; complementary to Random Forest; often faster to train |

### 9.2 Why Tree-Based Models Are Particularly Suitable

Tree-based models are well-suited for network intrusion detection on tabular data for several reasons:

- **No distributional assumptions**: Unlike parametric models, tree-based methods do not assume specific feature distributions, accommodating the often skewed and heavy-tailed distributions observed in network flow data.
- **Intrinsic feature selection**: Decision trees inherently select the most informative features at each split, providing robustness in high-dimensional settings.
- **Handling of mixed feature types**: Tree-based algorithms can process both numerical and categorical features without extensive preprocessing.
- **Feature importance extraction**: They provide built-in mechanisms for ranking feature importance, which supports cybersecurity interpretation.

### 9.3 Training Procedure

Each model was trained independently within its own preprocessing pipeline, ensuring that:

- Preprocessing parameters are model-specific and do not leak across experiments.
- Each pipeline is self-contained and can be independently serialised, loaded, and deployed.

---

## 10. Hyperparameter Tuning

### 10.1 Tuning Strategy

Hyperparameter optimisation was conducted using **`RandomizedSearchCV`**, which samples a fixed number of hyperparameter configurations from specified distributions and evaluates each via stratified cross-validation.

| Configuration | Value |
|---|---|
| **Search method** | Randomised search (20 iterations) |
| **Cross-validation** | 3-fold stratified CV |
| **Scoring metric** | `f1_macro` |
| **Models tuned** | Top 2 performers from initial evaluation |

### 10.2 Why F1 Macro for Scoring

The `f1_macro` metric computes the F1 score independently for each class and takes the unweighted mean. This ensures that the tuning process optimises for balanced performance across **all** attack types, preventing the search from favouring configurations that merely excel on majority classes.

### 10.3 Hyperparameters Explored

For **Random Forest** (and similarly structured tree-based ensembles), the following hyperparameters were explored:

- **`n_estimators`**: Number of trees in the ensemble (controls bias-variance trade-off).
- **`max_depth`**: Maximum depth of individual trees (controls model complexity and overfitting).
- **`min_samples_split`**: Minimum samples required to split an internal node.
- **`min_samples_leaf`**: Minimum samples required at a leaf node.
- **`max_features`**: Number of features considered at each split (introduces randomness).
- **`class_weight`**: Weighting strategy for class imbalance.

### 10.4 Why Tuning Is Not Performed on the Test Set

Hyperparameter tuning must use only training data (via cross-validation) to prevent **optimistic bias**. If the test set were used to select hyperparameters, the model would be indirectly fitted to test data, and the reported performance would no longer represent an unbiased estimate of generalisation ability. The test set is reserved exclusively for final, one-time evaluation.

---

## 11. Model Evaluation

### 11.1 Evaluation Metrics

Model performance was assessed using a suite of metrics chosen to reflect both overall and per-class detection capability:

#### Primary Metrics

- **Macro F1 Score**: The unweighted mean of per-class F1 scores. This metric treats all classes equally regardless of their frequency, making it the most appropriate primary metric for imbalanced multi-class classification. A model with high macro F1 reliably detects **all** attack types, including rare ones.
- **Weighted F1 Score**: The weighted mean of per-class F1 scores, where weights are proportional to class frequency. This provides a complementary perspective that accounts for class prevalence.
- **Per-class Recall**: The proportion of actual instances of each class that were correctly identified. In cybersecurity, recall is particularly critical—missed attacks (false negatives) can have severe consequences.

#### Secondary Metrics

- **Accuracy**: The overall proportion of correct predictions. While widely used, accuracy is reported as a secondary metric because it can be misleadingly high in imbalanced settings (e.g., a model predicting only the majority class may achieve >90% accuracy while failing to detect any attacks).

### 11.2 Detailed Evaluation Outputs

- **Classification Report**: Per-class precision, recall, and F1 score for every attack type, enabling identification of specific classes where the model underperforms.
- **Confusion Matrix**: A visual heatmap of predicted versus actual labels, revealing patterns of misclassification (e.g., which attack types are commonly confused with normal traffic).

---

## 12. Binary Evaluation

### 12.1 Secondary Interpretation

As a supplementary analysis, model predictions were re-interpreted through a **binary lens**:

- All legitimate IoT traffic classes were mapped to **"Normal"**.
- All attack categories were mapped to **"Malicious"**.

### 12.2 Binary Metrics

The following metrics were computed under this binary interpretation:

- **Binary Accuracy**: Overall correctness of the normal vs. malicious classification.
- **Precision**: Of all instances flagged as malicious, the proportion that were truly malicious (relevance of alerts).
- **Recall**: Of all truly malicious instances, the proportion that were correctly detected (detection rate).
- **F1 Score**: The harmonic mean of precision and recall, balancing alert quality and detection completeness.
- **Binary Confusion Matrix**: Visualisation of true positives, false positives, true negatives, and false negatives.

### 12.3 Clarification

It is important to note that the **primary task remains multi-class classification**. The binary evaluation provides an additional operational perspective—answering the high-level question of whether the system can reliably separate benign traffic from any form of attack—but does not replace the granular multi-class analysis.

---

## 13. Final Model Selection

### 13.1 Selection Criterion

The final model was selected based on the **highest macro F1 score** achieved after hyperparameter tuning. This criterion ensures that the chosen model provides the most balanced detection capability across all attack categories, including those with limited training samples.

### 13.2 Comparison Framework

The selection process involved:

1. **Baseline comparison**: Performance of all five untuned models was recorded as a reference.
2. **Tuned comparison**: The top-performing models were tuned, and their post-tuning metrics were compared against both each other and the untuned baselines.
3. **Improvement assessment**: The magnitude of improvement from tuning was quantified to determine whether the additional computational cost was justified.

### 13.3 Model Serialisation

The final selected model was saved as a **complete pipeline object** encapsulating both the preprocessing steps and the trained classifier. This ensures that deployed predictions are processed identically to training data without requiring separate preprocessing logic.

---

## 14. Feature Importance Analysis

### 14.1 Purpose

Feature importance analysis identifies which network flow attributes are most discriminative for distinguishing between traffic classes. This analysis serves dual purposes:

- **Model interpretability**: Understanding *what* the model has learned, which builds trust and enables validation by domain experts.
- **Cybersecurity insight**: Revealing which network characteristics are most indicative of specific attack types, informing the design of targeted detection rules or monitoring strategies.

### 14.2 Extraction Method

Feature importances were extracted from the tree-based model's **`feature_importances_`** attribute, which quantifies the contribution of each feature to reducing impurity (e.g., Gini impurity) across all trees in the ensemble.

### 14.3 Feature Name Mapping

Because the preprocessing pipeline transforms raw features (particularly through one-hot encoding of categorical variables), the importance values correspond to **transformed** feature names. The analysis correctly maps these back to their original feature names, ensuring that the reported importances are interpretable in the context of the original network flow attributes.

### 14.4 Visualisation

- **Top 10 Features**: A focused view of the most influential features for rapid interpretation.
- **Top 20 Features**: An extended view providing additional context for comprehensive analysis.

### 14.5 Cybersecurity Interpretation

The most important features typically correspond to network flow characteristics that differentiate attack traffic from normal communication—such as packet size statistics, flow duration, inter-arrival times, and protocol-specific counters. These findings can guide security analysts in prioritising which traffic attributes to monitor most closely.

---

## 15. Model Saving and Deployment

### 15.1 Artefact Persistence

The trained system is saved as two complementary artefacts:

| Artefact | Format | Contents |
|---|---|---|
| **Model pipeline** | `.joblib` (serialised via `joblib`) | Complete scikit-learn pipeline including fitted preprocessor and trained classifier |
| **Model metadata** | `.json` | Model name, expected feature names, training date, and performance metrics |

### 15.2 Why a Complete Pipeline Is Saved

Saving the entire pipeline—rather than just the model weights—ensures that:

- **Preprocessing is embedded**: The exact scaling parameters and encoding mappings are preserved, eliminating the risk of preprocessing mismatches at inference time.
- **Deployment is self-contained**: A single object handles the complete transformation from raw input to prediction.

### 15.3 Deployment Interface

A **Streamlit web application** (`app.py`) provides a user-friendly interface for interacting with the trained model. This interface enables non-technical users (e.g., security operations centre analysts) to submit network traffic data and receive classification results without requiring programming expertise.

---

## 16. Prediction Pipeline

### 16.1 Inference Workflow

The prediction pipeline follows a streamlined workflow for classifying new, unseen network traffic:

```
Load Saved Pipeline → Validate Input Columns → Preprocess (transform) → Predict → Output Results
```

### 16.2 Input Validation

Before prediction, the pipeline validates that the input data contains the expected feature columns. This safeguard prevents silent errors that could arise from missing or reordered features.

### 16.3 Outputs

Each prediction generates three outputs:

| Output | Description |
|---|---|
| **Predicted_Attack_Type** | The specific attack category (or normal) assigned by the classifier |
| **Traffic_Status** | A binary label—*Normal* or *Malicious*—derived from the predicted attack type |
| **Prediction_Probability** | The model's confidence in its prediction, expressed as the maximum class probability |

### 16.4 Operational Characteristics

- **No retraining required**: The saved pipeline applies the previously learned preprocessing transformations and model parameters, enabling immediate inference on new data.
- **Consistent preprocessing**: The same scaling, encoding, and feature selection applied during training are automatically applied to new inputs, ensuring consistency between training and inference environments.

---

## Methodology Flow Diagram

The complete methodology follows a sequential pipeline architecture, where each stage builds upon the outputs of its predecessor:

```
┌─────────────────────┐
│  Problem Definition │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Dataset Collection  │
│   (RT-IoT2022)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Data Understanding  │
│ (Features, Types,   │
│  Target Variable)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Data Preprocessing  │
│ (Missing Values,    │
│  Inf Handling, IDs) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Exploratory Data  │
│   Analysis (EDA)    │
│ (Distributions,     │
│  Correlations,      │
│  Imbalance Check)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Train/Test Split   │
│  (80/20 Stratified, │
│   random_state=42)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Class Imbalance    │
│  Handling           │
│  (class_weight=     │
│   "balanced")       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature             │
│ Transformation      │
│ (StandardScaler +   │
│  OneHotEncoder via  │
│  ColumnTransformer) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Training     │
│  (LR, DT, RF,      │
│   HGB, ET)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Hyperparameter     │
│  Tuning             │
│  (RandomizedSearch  │
│   CV, f1_macro)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Evaluation   │
│  (Macro F1, Recall, │
│   Confusion Matrix) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Final Model        │
│  Selection          │
│  (Best Macro F1)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Importance │
│  Analysis           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Saving &     │
│  Deployment         │
│  (joblib + JSON +   │
│   Streamlit)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Prediction         │
│  Pipeline           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Intrusion        │
│  Classification     │
│  (Normal/Malicious  │
│   + Attack Type)    │
└─────────────────────┘
```

---

## Summary

This methodology implements a rigorous, end-to-end machine learning pipeline for network intrusion detection in IoT environments. Key methodological principles upheld throughout include:

- **No data leakage**: Preprocessing is fitted exclusively on training data; the test set is used only for final evaluation.
- **Reproducibility**: Fixed random states and deterministic pipelines ensure consistent results across executions.
- **Class-aware evaluation**: Macro F1 scoring ensures that rare attack types receive equal importance in model selection and tuning.
- **Deployment readiness**: The complete pipeline is serialised as a self-contained artefact, supporting immediate deployment via the Streamlit web interface.

These design decisions collectively ensure that the system provides reliable, interpretable, and operationally useful network intrusion detection suitable for real-world IoT security monitoring.
