# AI-Based Early Warning System for Drug-Abuse Risk Assessment Using Explainable Machine Learning

**Jesu Melwin P M.¹, Dr. A. Maria Vinitha²**  
¹M.Sc. Student, Department of Computer Science, Loyola College, Chennai–600034, Tamil Nadu, India  
²Assistant Professor, Department of Computer Science, Loyola College, Chennai–600034, Tamil Nadu, India  
*Email:* jesumelwin@gmail.com; mvinitha@loyolacollege.edu  

---

## ABSTRACT

Drug abuse and substance-use disorders are escalating global public health crises that demand early, interpretable, and prevention-oriented screening mechanisms. Existing assessment paradigms heavily emphasize post-onset biochemical drug screening, clinical toxicology, or isolated psychological inventories. While valuable, these reactive methodologies fail to synthesize multi-systemic risk and protective indicators into an accessible, graduated early-warning framework prior to addiction onset. This paper introduces an explainable, leakage-free machine learning system designed to stratify drug abuse vulnerability across three operational tiers: **Low Risk**, **Moderate Risk**, and **High Risk**. 

Using a structured dataset of 5,000 subjects with 87 initial variables, a rigorous data leakage audit was conducted. Seven post-substance consumption markers and structurally incomplete features were eliminated, isolating 35 non-leaking predictors across psychological, behavioral, peer, familial, academic/occupational, and exposure domains. Data preprocessing was strictly contained within an 80:20 stratified split (4,000 training, 1,000 holdout test samples) utilizing median imputation, StandardScaler normalization, and one-hot encoding. Six supervised learning algorithms—Multiclass Logistic Regression, Support Vector Machine (SVM), Random Forest, XGBoost, LightGBM, and Decision Tree—were optimized via 5-fold Stratified Cross-Validation using `RandomizedSearchCV` maximizing Macro F1-score. 

On the untouched holdout test partition (1,000 samples: 400 Low Risk, 300 Moderate Risk, 300 High Risk), **Multiclass Logistic Regression** achieved the highest overall performance: **Accuracy of 70.50%**, **Macro F1-score of 70.01%**, **Macro Precision of 70.30%**, **Macro Recall of 69.94%**, **Balanced Accuracy of 69.94%**, and an **OvR ROC-AUC of 0.8631**, operating with ultra-low training (0.0525 s) and inference latency (0.0080 s). Confusion matrix inspection demonstrated high clinical screening safety: only 6 out of 300 true High-Risk individuals (2.0%) were erroneously classified as Low Risk. Global and local feature attributions were extracted using Tree/Kernel SHAP (SHapley Additive exPlanations), quantifying the primary drivers of vulnerability (Previous Substance Exposure: mean |SHAP| = 0.403; Peer Substance Exposure: 0.232; Depression Score: 0.219; Stress Level: 0.219; Impulsivity: 0.212) alongside protective buffers (Family Support: 0.115; Emotional Regulation: 0.103; Social Support: 0.100). The validated pipeline is deployed as an interactive, human-in-the-loop decision-support system via Streamlit, offering interpretable risk scoring to empower early psychosocial intervention rather than deterministic diagnosis.

**Keywords:** Drug Abuse Risk Assessment, Machine Learning, Early Warning System, Multiclass Classification, Explainable AI, SHAP, Feature Selection, Preventive Decision Support.

---

## 1. INTRODUCTION

Substance use disorders (SUD) and illicit drug abuse represent multifaceted biopsychosocial pathologies that exert severe burdens on global healthcare infrastructures, family ecosystems, and socioeconomic productivity. Epidemiological studies on adolescent and young-adult behavioral health indicate that drug vulnerability is governed by complex, non-linear interactions across personal, familial, peer, and environmental spheres (Brook et al., 2014; Nawi et al., 2021). Factors such as trait impulsivity, emotional dysregulation, chronic stress, familial strife, peer substance affiliations, and school disengagement interact dynamically over developmental trajectories.

Crucially, an operational distinction must be established between **substance detection** and **substance abuse risk prediction**. Standard forensic and clinical toxicological assays (urinalysis, gas chromatography-mass spectrometry, electrochemical biosensors) are designed to identify active chemical analytes or metabolites in biofluids post-ingestion (Habibi et al., 2024; Malar et al., 2026). Similarly, neuroimaging, electroencephalographic (EEG) profiling, and wearable digital biomarkers capture immediate neurophysiological intoxication or craving states (Wu et al., 2025; Li et al., 2024). In contrast, preventive public health demands an accessible, questionnaire-driven computational paradigm that operates *upstream*—identifying vulnerable individuals prior to the onset of severe physiological dependence or clinical addiction.

Supervised machine learning (ML) provides a compelling framework to synthesize heterogeneous psychosocial determinants into coherent probabilistic risk estimates. However, the application of predictive analytics to substance vulnerability faces critical methodological pitfalls: (1) rampant data leakage through the inadvertent inclusion of post-outcome consumption variables (e.g., substance craving, frequency of use, withdrawal symptoms), (2) algorithmic opacity that alienates clinical practitioners, and (3) inappropriate evaluation metrics that mask minority class misclassifications.

To resolve these challenges, this study presents an end-to-end, leakage-free, explainable machine-learning early warning framework. The primary contributions of this work include:
1. **Methodological Rigor and Leakage Elimination:** Implementation of a strict two-stage feature audit isolating 35 psychosocial predictors and guaranteeing zero data leakage across preprocessing, imputation, scaling, and hyperparameter tuning.
2. **Comprehensive Empirical Benchmarking:** Systematic comparison of six supervised algorithms (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, Decision Tree) across 8 validation metrics, detailing training latency, inference speed, and complete multiclass confusion matrices.
3. **Model Interpretability via SHAP:** Granular decomposition of feature attributions utilizing SHAP beeswarm plots and mean absolute Shapley values, distinguishing between risk-amplifying stressors and protective psychosocial buffers.
4. **Interactive Translational Deployment:** Encapsulation of the trained model within a responsive, production-ready Streamlit decision-support application facilitating questionnaire-based screening and real-time visual explanations.

---

## 2. PROBLEM STATEMENT

Conventional screening mechanisms for substance abuse risk are predominantly bifurcated into invasive laboratory assays or qualitative, time-intensive clinical interviews. Structured psychometric instruments (such as the EU-Dap questionnaire) capture developmental vulnerability but frequently rely on rigid, unweighted composite scoring systems that fail to generalize across non-linear feature interactions (Ramírez et al., 2021). Furthermore, many published predictive models suffer from methodological contamination by incorporating variables that directly reflect active substance dependence (e.g., tolerance, withdrawal, rehabilitation history), artificially inflating performance while rendering the model useless for true pre-onset early warning.

The research problem investigated herein is the formulation, optimization, and empirical validation of an interpretable, non-invasive multiclass machine learning framework. The system must accurately stratify individuals into three actionable categories—**Low Risk**, **Moderate Risk**, and **High Risk**—relying solely on accessible, non-leaking psychosocial and lifestyle predictors, while providing mathematically grounded explanations to support timely preventive intervention.

---

## 3. RESEARCH OBJECTIVES

The specific objectives guiding this investigation are:
1. To identify, audit, and clean a multidimensional dataset comprising psychological, behavioral, familial, social, lifestyle, and historical exposure factors.
2. To isolate and eliminate structural data leakage, ensuring no post-substance consumption indicators contaminate the feature space.
3. To construct a hermetic preprocessing pipeline executing median numerical imputation, most-frequent categorical imputation, z-score standardization, and one-hot encoding strictly fitted on training splits.
4. To train and optimize six candidate machine learning models using 5-fold Stratified Cross-Validation and randomized hyperparameter search.
5. To evaluate models across holistic performance criteria including Accuracy, Macro Precision, Macro Recall, Macro F1-score, Balanced Accuracy, ROC-AUC, and computational execution time.
6. To perform full multiclass error decomposition via confusion matrix analysis, evaluating critical safety boundaries between adjacent and polar risk classes.
7. To extract global and local feature explanations using SHAP, elucidating risk drivers and protective mechanisms.
8. To deploy the optimal pipeline into an interactive Streamlit graphical interface for practical decision support.

---

## 4. LITERATURE REVIEW

### 4.1 Traditional Substance Detection and Psychometric Screening
Biochemical and electrochemical detection systems provide gold-standard verification of drug ingestion. Habibi et al. (2024) developed an electrochemical sensor paired with partial least squares regression to quantify morphine and methadone in urinary biofluids. Malar et al. (2026) reviewed AI-driven microfluidic and biosensor technologies for real-time analyte tracking. Although scientifically robust, these tools require specialized infrastructure, are inherently reactive, and cannot predict vulnerability in individuals who have not yet initiated regular substance use.

In the psychosocial domain, structured questionnaires such as the European Drug Addiction Prevention (EU-Dap) survey have been validated across adolescent cohorts (Ramírez et al., 2021). These surveys highlight school bonding, normative beliefs, and peer perceptions as core predictive axes. However, traditional psychometric evaluations lack dynamic scoring mechanisms capable of handling complex multivariable non-linearities.

### 4.2 Machine Learning in Substance Abuse Analytics
A broad spectrum of machine learning algorithms has been explored for substance abuse risk stratification. Han et al. (2019) utilized tree-based ensembles to predict adolescent opioid misuse across national survey datasets, reporting that behavioral delinquency and mental distress were primary indicators. Wadekar (2020) applied psychosocial modeling to adolescent substance use, demonstrating the utility of Random Forests. Islam et al. (2021) investigated machine learning techniques for substance vulnerability, noting that class imbalance and feature redundancy frequently degrade model reliability. Recent studies by Ünlü & Subaşı (2024) and Kim et al. (2024) underscored that ensemble complexity does not invariably outperform regularized linear models when predictors are strongly collinear and feature spaces are standardized.

### 4.3 Physiological, Wearable, and Digital Biomarkers
Recent paradigms have extended into continuous physiological tracking. Wu et al. (2025) proposed DSCnet, utilizing hybrid EEG feature representations to achieve 85.11% accuracy in classifying established drug addiction. In naturalistic environments, Li et al. (2024) evaluated Fitbit biosensors combined with convolutional neural networks, achieving an ROC-AUC of 0.729 for craving detection. Jeyadevan & Grigg (2024) provided a scoping review of wearable biosensors, highlighting hardware variability and patient compliance as persistent limitations. These digital modalities complement early warning frameworks but remain hardware-dependent and clinically targeted.

### 4.4 AI-Driven Recovery and Clinical Decision Support
Beyond initial detection, digital platforms such as the Behaivior Recovery system integrate mobile ecological momentary assessments (EMA) with predictive algorithms to forecast relapse among veterans (Förster et al., 2025). Khakpaki & Sepehri (2025) synthesized the ethical imperative of Explainable AI (XAI) in behavioral healthcare, emphasizing that black-box outputs hinder clinical adoption and risk perpetuating algorithmic stigma.

### 4.5 Synthesis and Research Gap
Existing literature demonstrates a distinct dichotomy: studies either concentrate on biological substance detection post-ingestion or employ deep learning on specialized physiological signals. A significant gap persists in establishing a standardized, leakage-free, open-access framework that synthesizes multidomain questionnaire indicators into a graduated 3-class risk stratification with rigorous SHAP-based explainability and production-ready deployment.

---

## 5. COMPARATIVE ANALYSIS OF EXISTING METHODS

Table 1 summarizes representative methodologies in substance abuse analytics, juxtaposing them against the proposed framework.

### Table 1: Comparative analysis of representative approaches in substance abuse analytics.
| Study | Year | Modality / Data Source | Methodology / Target | Key Strengths | Critical Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nawi et al.** | 2021 | Systematic review (Adolescents) | Psychosocial risk/protective factor mapping | Synthesizes multidomain ecological factors | Qualitative synthesis; lacks predictive modeling |
| **Ramírez et al.** | 2021 | Questionnaire (N=2,261 Chilean youth) | EU-Dap psychometric validation | Validated developmental scales | Rigid linear scoring; population-specific |
| **Wu et al.** | 2025 | 64-channel resting-state EEG | DSCnet deep learning / Addiction detection | Captures neurophysiological biomarkers | High equipment cost; detects current addiction only |
| **Habibi et al.** | 2024 | Urinary biofluids / Voltammetric sensors | Partial Least Squares / Morphine & methadone | Precise chemical quantification | Purely reactive post-consumption assay |
| **Li et al.** | 2024 | Wearable biosensors (Fitbit, N=9) | CNN / Craving & momentary use tracking | Continuous passive monitoring | Small sample size; wearable compliance barriers |
| **Förster et al.**| 2025 | App-based EMA & wearable telemetry | Predictive alerting / Veteran recovery support | Real-time relapse risk mitigation | Focuses on recovery/relapse rather than early prevention |
| **Present Study**| 2026 | Structured multidomain dataset (N=5,000) | Leakage-free ML + SHAP / 3-Class Risk | Non-invasive, interpretable, zero-leakage pipeline | Requires prospective multi-center validation |

### Table 2: Comparison of operational modalities across assessment targets.
| Assessment Modality | Primary Input Features | Target Outcome | Specialized Hardware | Upstream Early Warning | Native Interpretability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Biochemical Testing** | Blood, urine, oral fluid | Active metabolite presence | Mass spectrometer / Test strips | No (Post-consumption) | Low (Binary assay report) |
| **Neuroimaging / EEG** | Brainwave oscillations | Addictive neural circuitry | EEG cap / Amplifiers | No (Post-onset pathology) | Low (Black-box deep neural net) |
| **Digital Wearables** | Heart rate variability, EDA | Momentary craving / Relapse | Smartwatch / Biosensors | Moderate (Short-term warning)| Moderate (Time-series features) |
| **Proposed Framework** | 35 Psychosocial indicators | 3-Tier Risk Stratification | None (Standard web browser) | **Yes (Upstream prevention)** | **High (SHAP Attribution Engine)**|

---

## 6. PROPOSED METHODOLOGY

### 6.1 Data Inspection and Cleaning
The dataset comprises 5,000 complete participant records across 87 raw attributes. An exploratory audit confirmed zero corrupted entries and consistent data types. The operational target variable, `Drug_Abuse_Risk`, is partitioned into three balanced, clinically meaningful classes:
* **Low Risk:** 2,000 observations (40.0%)
* **Moderate Risk:** 1,500 observations (30.0%)
* **High Risk:** 1,500 observations (30.0%)

### 6.2 Leakage Prevention and Feature Audit
A critical flaw in historical risk models is the presence of *target leakage*. Including variables that measure established substance use invalidates a model's utility as an early warning screener. To enforce rigorous methodological integrity, a strict exclusion protocol was implemented:
1. **Direct Post-Consumption Variables Excluded:** `Type_of_Substance_Used`, `Frequency_of_Use`, `Duration_of_Use`, `Craving_Level`, `Withdrawal_Symptoms`, and `Previous_Rehabilitation_Counselling`.
2. **Exposure Severity Leakage Excluded:** `Substance_Exposure_Frequency` and `Polysubstance_Exposure`.
3. **Structural Missingness Excluded:** `Age_First_Substance_Exposure` was eliminated because missing values structurally indicated non-exposure, thereby introducing latent leakage.

Following exclusion, 35 clean, upstream predictors spanning six domains were retained.

### Table 3: Summary of the 35 retained non-leaking predictors categorized by domain.
| Domain | Number of Features | Retained Predictor Variables |
| :--- | :--- | :--- |
| **Psychological** | 10 | Depression Score, Anxiety Level, Stress Level, Psychological Distress, Impulsivity, Self-Esteem, Emotional Regulation, Hopelessness Level, Loneliness Level, Self-Control |
| **Social & Peer** | 5 | Peer Influence, Peer Substance Exposure, Social Isolation, Social Support, Social Pressure |
| **Familial** | 3 | Family Conflict, Family Support, Family Stability |
| **Behavioral** | 5 | Behavioural Problems, Decision-Making Control, Substance Use Intention, Attitude Toward Substance Use, School/Work Absenteeism |
| **Lifestyle** | 4 | Sleep Duration, Sleep Quality, Screen Time, Night-Time Device Usage |
| **Academic & Occupational** | 7 | Academic Performance, Academic/Work Stress, Workload Level, Job Satisfaction, Financial Stress, Childhood Adversity, Previous Substance Exposure |

### 6.3 Preprocessing Pipeline
To prevent data snooping, all preprocessing estimators were embedded within a unified `scikit-learn` pipeline and fitted exclusively on the 4,000 training observations:
* **Numerical Imputation & Scaling:** Missing numeric entries were imputed using training medians and standardized via z-score normalization:
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
* **Categorical Imputation & Encoding:** Categorical attributes were imputed using the most frequent training category and converted into binary columns via `OneHotEncoder(handle_unknown='ignore')`.

---

## 7. MACHINE-LEARNING MODELS

Six supervised classification algorithms spanning diverse architectural paradigms were trained and evaluated:

### 7.1 Multiclass Logistic Regression
Multiclass Logistic Regression applies a linear combination of input features mapped to normalized class probabilities using the softmax activation function:
$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
The model was trained using the `lbfgs` solver with $L_2$ regularization. Its high convexity and resistance to overfitting make it exceptionally reliable in balanced, standardized psychosocial feature spaces.

### 7.2 Support Vector Machine (SVM)
The Support Vector Machine constructs hyperplanes in high-dimensional feature space that maximize the functional margin between classes. For multiclass discrimination, a One-vs-Rest (OvR) formulation was utilized with a linear kernel:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
Standardization ensures that distance metrics are not biased by feature variances.

### 7.3 Decision Tree and Random Forest
Decision Trees recursively partition feature space based on Gini impurity reduction. To overcome tree instability and variance, Random Forest aggregates an ensemble of $B = 200$ decorrelated bootstrap decision trees, selecting random subsets of features at each split:
$$\hat{y} = \text{mode} \left\{ T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x}) \right\}$$

### 7.4 Gradient Boosted Decision Trees (XGBoost and LightGBM)
Extreme Gradient Boosting (XGBoost) and Light Gradient Boosting Machine (LightGBM) sequentially build shallow regression trees that fit the negative gradients of the multi-class log-loss function. Both frameworks incorporate shrinkage ($\eta = 0.1$) and feature subsampling (`colsample_bytree`) to limit over-specialization on noisy behavioral survey responses.

### Table 4: Hyperparameter optimization search space and optimal configurations.
| Algorithm | Optimized Hyperparameters | Best Parameter Configuration |
| :--- | :--- | :--- |
| **Logistic Regression** | Solver, Regularization Strength ($C$), Class Weight | `{'solver': 'lbfgs', 'C': 10.0, 'class_weight': 'balanced'}` |
| **SVM** | Kernel, Cost Parameter ($C$), Gamma | `{'kernel': 'linear', 'C': 1.0, 'gamma': 'scale', 'class_weight': None}` |
| **Random Forest** | Estimators, Max Depth, Min Samples Split, Leaf Size | `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 2, 'class_weight': 'balanced'}` |
| **XGBoost** | Estimators, Max Depth, Learning Rate, Subsample | `{'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 1.0}` |
| **LightGBM** | Estimators, Num Leaves, Max Depth, Subsample | `{'n_estimators': 100, 'num_leaves': 31, 'max_depth': 10, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 0.8}` |
| **Decision Tree** | Criterion, Max Depth, Min Samples Split, Leaf Size | `{'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 1, 'class_weight': 'balanced'}` |

---

## 8. MODEL TRAINING AND EVALUATION

### 8.1 Training and Cross-Validation Protocol
The dataset of 5,000 instances was partitioned into a locked 80% training set ($N_{\text{train}} = 4,000$) and a 20% holdout test set ($N_{\text{test}} = 1,000$) using stratified sampling to preserve the 40:30:30 class ratio. Hyperparameter tuning was executed strictly on the training partition via 5-Fold Stratified Cross-Validation using `RandomizedSearchCV` across 50 iterations per model family.

### 8.2 Evaluation Metrics
To guarantee robust assessment across asymmetric error costs, models were evaluated on the independent 1,000-sample test partition across multiple metrics:
1. **Overall Accuracy:** Ratio of correct classifications to total test instances.
2. **Macro-Averaged F1-Score:** Unweighted mean of class-specific F1-scores, ensuring minority risk classes are evaluated equally:
   $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
3. **Balanced Accuracy:** Average recall achieved across each individual class.
4. **Multiclass ROC-AUC:** Macro-averaged Area Under the Receiver Operating Characteristic curve computed using the One-vs-Rest (OvR) formulation.
5. **Computational Latency:** Wall-clock training duration (s) and single-batch test prediction latency (s).

---

## 9. EXPERIMENTAL RESULTS AND DISCUSSION

### 9.1 Overall Model Performance Comparison
Table 5 presents the complete empirical benchmark across all six algorithms on the unseen 1,000-sample test set.

### Table 5: Comprehensive holdout test performance benchmark across all six machine learning models.
| Model Family | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Accuracy | ROC-AUC (OvR) | Training Time (s) | Prediction Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.7050** | **0.7030** | **0.6994** | **0.7001** | **0.6994** | **0.8631** | **0.0525** | **0.0080** |
| **Support Vector Machine**| 0.6990 | 0.6873 | 0.6878 | 0.6875 | 0.6878 | 0.8619 | 3.3507 | 0.1014 |
| **Random Forest** | 0.6820 | 0.6710 | 0.6725 | 0.6717 | 0.6725 | 0.8471 | 0.9548 | 0.0531 |
| **XGBoost** | 0.6770 | 0.6578 | 0.6614 | 0.6584 | 0.6614 | 0.8535 | 0.2096 | 0.0130 |
| **LightGBM** | 0.6760 | 0.6582 | 0.6619 | 0.6593 | 0.6619 | 0.8415 | 0.1411 | 0.0150 |
| **Decision Tree** | 0.5790 | 0.5831 | 0.5739 | 0.5762 | 0.5739 | 0.7037 | 0.0531 | 0.0080 |

### Key Experimental Findings:
1. **Linear Superiority in High-Dimensional Psychosocial Space:** **Multiclass Logistic Regression achieved the highest performance across every primary metric**, reaching **70.50% Accuracy**, **70.01% Macro F1**, and **0.8631 ROC-AUC**. Linear SVM followed closely behind (69.90% Accuracy, 0.8619 ROC-AUC).
2. **Ensemble Over-Fitting to Survey Variance:** Complex non-linear gradient boosting ensembles (XGBoost: 67.70% Accuracy; LightGBM: 67.60% Accuracy) and Random Forest (68.20% Accuracy) underperformed the linear models. This confirms that when psychosocial survey indicators exhibit additive risk profiles and continuous covariance, regularized linear decision boundaries prevent overfitting to idiosyncratic survey noise.
3. **Computational Efficiency:** Logistic Regression trained in just **0.0525 seconds**—over 63 times faster than linear SVM (3.3507 s)—and achieved inference in **8.0 milliseconds**, establishing it as the ideal candidate for lightweight web deployment.

---

### 9.2 Granular Multiclass Confusion Matrix Analysis
To rigorously inspect class-specific performance, confusion matrices were extracted across all 1,000 test observations (comprising 300 High Risk, 400 Low Risk, and 300 Moderate Risk actual cases).

### Table 6: Empirical Multiclass Confusion Matrices across all six evaluated models ($N_{\text{test}} = 1,000$).
| Model | True Class | Pred: High Risk | Pred: Low Risk | Pred: Moderate Risk | Class Recall | Class Precision | Class F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **High Risk** (300) | **226** | **6** | 68 | **75.33%** | **75.33%** | **0.753** |
| *(Optimal Pipeline)* | **Low Risk** (400) | 8 | **302** | 90 | **75.50%** | **82.74%** | **0.789** |
| | **Moderate Risk** (300)| 66 | 57 | **177** | **59.00%** | **52.84%** | **0.557** |
| **Support Vector Machine**| High Risk (300) | 226 | 10 | 64 | 75.33% | 75.59% | 0.755 |
| | Low Risk (400) | 7 | 320 | 73 | 80.00% | 77.86% | 0.789 |
| | Moderate Risk (300)| 66 | 81 | 153 | 51.00% | 52.76% | 0.519 |
| **Random Forest** | High Risk (300) | 224 | 14 | 62 | 74.67% | 72.96% | 0.738 |
| | Low Risk (400) | 11 | 307 | 82 | 76.75% | 77.14% | 0.769 |
| | Moderate Risk (300)| 72 | 77 | 151 | 50.33% | 51.19% | 0.508 |
| **XGBoost** | High Risk (300) | 218 | 14 | 68 | 72.67% | 71.24% | 0.719 |
| | Low Risk (400) | 12 | 327 | 61 | 81.75% | 75.52% | 0.785 |
| | Moderate Risk (300)| 76 | 92 | 132 | 44.00% | 50.57% | 0.471 |
| **LightGBM** | High Risk (300) | 219 | 14 | 67 | 73.00% | 70.65% | 0.718 |
| | Low Risk (400) | 13 | 321 | 66 | 80.25% | 76.25% | 0.782 |
| | Moderate Risk (300)| 78 | 86 | 136 | 45.33% | 50.56% | 0.478 |
| **Decision Tree** | High Risk (300) | 184 | 20 | 96 | 61.33% | 62.37% | 0.618 |
| | Low Risk (400) | 33 | 250 | 117 | 62.50% | 72.05% | 0.669 |
| | Moderate Risk (300)| 78 | 77 | 145 | 48.33% | 40.50% | 0.441 |

### Critical Safety and Screening Implications:
1. **Exceptional Containment of Severe False Negatives:** In a clinical early warning context, the most hazardous error is misclassifying a **High-Risk** individual as **Low Risk** (failing to trigger preventive support). In Logistic Regression, **only 6 out of 300 High-Risk cases (2.0%)** were erroneously assigned to Low Risk. Conversely, only 8 out of 400 Low-Risk individuals (2.0%) were mislabeled as High Risk.
2. **Boundary Nature of Moderate Risk:** Across all models, the Moderate Risk class displays lower precision (52.84% in LR) and recall (59.00% in LR). The confusion matrix demonstrates that errors are concentrated almost exclusively between *adjacent* boundaries (68 High-Risk cases classified as Moderate; 90 Low-Risk cases classified as Moderate). This reflects clinical reality: intermediate behavioral risk exists along a fluid continuum rather than within discrete boundaries.

---

### 9.3 Explainability Analysis via SHAP (SHapley Additive exPlanations)
To unpack the model's internal decision logic, SHAP values were calculated across all test instances.

### Table 7: Quantitative feature importance ranking of the top 15 predictors from global SHAP analysis.
| Rank | Feature Name | Domain | Mean Absolute SHAP Value ($E[\|\phi_i\|]$) | Primary Direction of Risk Impact |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `Previous_Substance_Exposure` | Historical / Exposure | **0.4026** | Binary catalyst; affirmative exposure sharply drives High Risk |
| **2** | `Peer_Substance_Exposure` | Social / Peer | **0.2324** | Direct risk driver; high peer use elevates predicted risk |
| **3** | `Depression_Score` | Psychological | **0.2193** | Escalating risk driver; elevated scores increase risk |
| **4** | `Stress_Level` | Psychological | **0.2187** | Chronic stress strongly amplifies vulnerability |
| **5** | `Impulsivity` | Psychological / Behavioral | **0.2125** | Behavioral accelerator; high impulsivity drives risk |
| **6** | `Psychological_Distress` | Psychological | **0.1929** | Global distress index positively correlates with risk |
| **7** | `Anxiety_Level` | Psychological | **0.1770** | Moderate-to-strong positive risk contributor |
| **8** | `Peer_Influence` | Social / Peer | **0.1533** | Susceptibility to peer norms drives risk upwards |
| **9** | `Family_Conflict` | Familial | **0.1512** | Chronic household disharmony elevates risk |
| **10** | `Social_Isolation` | Social / Peer | **0.1256** | Lack of prosocial integration increases vulnerability |
| **11** | `Family_Support` | Familial | **0.1147** | **Protective buffer;** high support suppresses risk |
| **12** | `Attitude_Toward_Substance_Use`| Behavioral | **0.1116** | Permissive/approving attitudes elevate risk |
| **13** | `Childhood_Adversity` | Historical / Environmental | **0.1100** | Cumulative developmental trauma drives risk |
| **14** | `Emotional_Regulation` | Psychological | **0.1031** | **Protective buffer;** cognitive appraisal buffers risk |
| **15** | `Social_Support` | Social / Peer | **0.0999** | **Protective buffer;** community connection suppresses risk |

### Bi-Directional Beeswarm Plot Interpretation:
Analysis of the global SHAP beeswarm distribution provides vital translational insights into how feature values modulate prediction:
* **Acute Vulnerability Accelerators (Positive SHAP Drivers):** High values (red points) for `Peer_Substance_Exposure`, `Stress_Level`, `Impulsivity`, `Depression_Score`, `Anxiety_Level`, and `Family_Conflict` uniformly extend along the positive SHAP axis, strongly pushing predictions toward **High Risk**.
* **Protective Resilience Buffers (Negative SHAP Drivers):** High values for `Family_Support`, `Emotional_Regulation`, `Social_Support`, `Sleep_Duration`, and `Sleep_Quality` project into the negative SHAP zone. This confirms empirically that robust family and social networks actively buffer individuals against psychological distress.
* **Non-Exposure Protective Shift:** The absence of prior substance exposure (`Previous_Substance_Exposure_No = 1`) delivers the single strongest negative SHAP impulse ($-0.654$), serving as the model's most potent anchor against high-risk classification.

---

## 10. SYSTEM ARCHITECTURE AND STREAMLIT DEPLOYMENT

The validated predictive pipeline was converted into a deployable software architecture and encapsulated within an interactive Streamlit application.

```
+-------------------------------------------------------------------------------+
|                       SYSTEM DEPLOYMENT ARCHITECTURE                          |
+-------------------------------------------------------------------------------+
|                           USER / CLINICIAN CLIENT                             |
|      Interactive Streamlit Web GUI (Accessible via Desktop / Tablet / Mobile) |
+---------------------------------------+---------------------------------------+
                                        |
                                        v  (HTTP POST / Session State)
+---------------------------------------+---------------------------------------+
|                    APPLICATION CONTROLLER & VALIDATOR                         |
|   - 35 Structured Input Widgets grouped into 6 Clinical Accordion Panels      |
|   - Input Integrity Guard (Range validation, Type coercion, Null checking)   |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+---------------------------------------+---------------------------------------+
|              SERIALIZED INFERENCE ENGINE (best_model_pipeline.pkl)            |
|   +-----------------------------------------------------------------------+   |
|   | PREPROCESSING STEP:                                                   |   |
|   | - ColumnTransformer (StandardScaler on Numeric, OneHot on Categoric)  |   |
|   +-----------------------------------------------------------------------+   |
|   | PREDICTIVE CLASSIFIER:                                                |   |
|   | - Multinomial Logistic Regression Model (C=10.0, lbfgs, balanced)     |   |
|   +-----------------------------------------------------------------------+   |
|   | LOCAL EXPLAINABILITY MODULE:                                          |   |
|   | - SHAP Attribution Generator (Patient-specific waterfall decomposition)|  |
|   +-----------------------------------------------------------------------+   |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+---------------------------------------+---------------------------------------+
|                       TRANSLATIONAL OUTPUT INTERFACE                          |
|   - Graduated Risk Level: [LOW RISK | MODERATE RISK | HIGH RISK]              |
|   - Calibrated Class Probability Distribution Bar Charts                      |
|   - Top 5 Personalized Driving Stressors vs Protective Buffers                |
|   - Actionable Psychosocial Recommendations & Healthcare Referral Protocols   |
+-------------------------------------------------------------------------------+
```

### Application Features:
1. **Interactive Questionnaire Interface:** Gathers 35 validated indicators categorized into logical clinical groupings (Psychological Health, Social/Peer Dynamics, Family Climate, Behavioral Patterns, Lifestyle Habits, and Academic/Occupational Factors).
2. **Transparent Risk Stratification:** Provides real-time probabilities across Low, Moderate, and High Risk tiers, avoiding binary or stigmatizing labels.
3. **Integrated Explainability & Actionable Advice:** Generates immediate feature-level explanations highlighting key risk drivers and protective assets, paired with personalized recommendations for counseling, stress mitigation, and social support.

---

## 11. ETHICAL, LEGAL, AND SOCIETAL CONSIDERATIONS

### 11.1 Privacy, Data Sovereignty, and Security
Deploying automated risk assessment systems in educational or community environments involves processing sensitive personal data. Implementation must adhere to strict regulatory standards (e.g., GDPR, HIPAA, and national digital personal data protection acts). All inputs should be processed ephemerally or stored using salted cryptographic hashing with strict role-based access control.

### 11.2 Mitigation of Algorithmic Stigmatization
A persistent danger in behavioral health AI is deterministic labeling. Labeling individuals as "addicts" or "destined for drug abuse" produces self-fulfilling prophecies and institutional bias. This system explicitly frames its output as a **probabilistic early-warning vulnerability index** designed to guide supportive counseling, strictly prohibiting punitive or disciplinary applications.

### 11.3 Human-in-the-Loop Clinical Governance
The system is explicitly engineered as an assistive decision-support prototype. It does not replace medical practitioners, clinical psychologists, or addiction specialists. High-risk alerts must serve solely as prompts for professional human evaluation.

---

## 12. LIMITATIONS

While this study establishes an empirically validated, leakage-free framework, several limitations must be acknowledged:
1. **Cross-Sectional Dataset:** The evaluation is based on a structured cohort of 5,000 observations. Cross-sectional data capture static associations rather than longitudinal causal progressions.
2. **Absence of Prospective Longitudinal Validation:** The model's capacity to forecast multi-year progression from Moderate to High Risk requires prospective tracking across diverse external demographic cohorts.
3. **Survey Self-Report Vulnerability:** Questionnaires are inherently subject to social desirability bias, recall inaccuracies, and deliberate under-reporting of illicit behaviors.
4. **Intermediate Boundary Overlap:** Classification of Moderate Risk exhibits moderate precision (52.84%), reflecting the natural continuum of psychosocial distress and pointing to the need for continuous severity indexing in future iterations.

---

## 13. FUTURE WORK

Future research directions will expand upon this framework along several key dimensions:
1. **Multi-Center Longitudinal Cohort Studies:** Validating the 35-feature pipeline across independent geographic, socio-demographic, and clinical populations to verify transportability.
2. **Multimodal Data Integration:** Combining self-reported psychosocial inventories with passive digital phenotyping (e.g., sleep regularity from consumer wearables, circadian screen-interaction patterns) under strict patient consent.
3. **Threshold Optimization & Cost-Sensitive Learning:** Calibrating decision thresholds to explicitly penalize false negatives in high-risk categories, further tightening screening safety.
4. **Federated and Privacy-Preserving Learning:** Exploring decentralized federated learning architectures to enable model training across healthcare networks without centralizing sensitive patient records.

---

## 14. CONCLUSION

This study demonstrates that an explainable, leakage-free machine learning early warning framework can accurately stratify drug abuse vulnerability using non-invasive psychosocial predictors. By rigorously auditing the feature space and eliminating post-consumption indicators, the system ensures true upstream screening capability. 

Empirical benchmarks on a 1,000-subject holdout test set established **Multiclass Logistic Regression** as the optimal model, achieving **70.50% Accuracy**, **70.01% Macro F1-score**, and **0.8631 ROC-AUC**, outperforming complex ensemble and boosting methods while operating with sub-10ms inference latency. Crucially, the model maintained an exceptional safety profile, yielding only a 2.0% critical false negative rate on high-risk individuals. Global SHAP explainability revealed that previous substance exposure, peer exposure, depression, stress, and impulsivity are the predominant drivers of risk, while family support, emotional regulation, and social support act as vital protective buffers. 

Integrated into an interactive Streamlit application, this research provides an accessible, transparent, and ethically grounded decision-support tool capable of facilitating early awareness, timely counseling referrals, and targeted psychosocial interventions before severe drug dependency occurs.

---

## REFERENCES

1. F. Bulut and İ. Ö. Bucak, "An urgent precaution system to detect students at risk of substance abuse through classification algorithms," *Turkish Journal of Electrical Engineering & Computer Sciences*, vol. 22, no. 5, pp. 1208–1220, 2014. doi: 10.3906/elk-1208-60.
2. P. Nath, S. Kilam, and A. Swetapadma, "A machine learning approach to predict volatile substance abuse for drug risk analysis," in *2017 2nd International Conference on Recent Trends in Electronics, Information & Communication Technology (RTEICT)*, 2017, pp. 823–827. doi: 10.1109/ICRCICN.2017.8234516.
3. Z. Qiao, T. Chai, Q. Zhang, X. Zhou, and Z. Chu, "Predicting potential drug abusers using machine learning techniques," in *2019 IEEE 4th International Conference on Cloud Computing and Big Data Analytics (ICCCBDA)*, 2019, pp. 490–494. doi: 10.1109/ICIIBMS46890.2019.8991550.
4. Y. Jing, Z. Hu, P. Fan, Y. Xue, L. Wang, R. Tarter, L. Kirisci, J. Wang, M. Vanyukov, and X. Xie, "Analysis of substance use and its outcomes by machine learning I. Childhood Evaluation of Liability to Substance Use Disorder," *Drug and Alcohol Dependence*, vol. 204, p. 107605, 2019. doi: 10.1016/j.drugalcdep.2019.107605.
5. D.-H. Han, S. Lee, and D. Seo, "Using machine learning to predict opioid misuse among U.S. adolescents," *Preventive Medicine*, vol. 130, p. 105886, 2020. doi: 10.1016/j.ypmed.2019.105886.
6. A. S. Wadekar, "A Psychosocial Approach to Predicting Substance Use Disorder (SUD) Among Adolescents," in *2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)*, 2020, pp. 784–788. doi: 10.1109/ASONAM49781.2020.9381378.
7. U. I. Islam, I. H. Sarker, E. Haque, and M. M. Hoque, "Predicting Individual Substance Abuse Vulnerability using Machine Learning Techniques," in *Security, Privacy, and Anonymity in Computation, Communication, and Storage*, Cham: Springer, 2021, pp. 505–516. doi: 10.1007/978-3-030-73050-5_42.
8. S. Kim, H. Kim, S. Kim, et al., "Machine Learning-Based Prediction of Substance Use in Adolescents in Three Independent Worldwide Cohorts: Algorithm Development and Validation Study," *Journal of Medical Internet Research*, vol. 26, p. e62805, 2024. doi: 10.2196/62805.
9. A. Ünlü and A. Subaşı, "Substance use prediction using artificial intelligence techniques," *Journal of Computational Social Science*, vol. 7, pp. 356–375, 2024. doi: 10.1007/s42001-024-00356-6.
10. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, vol. 30, 2017. doi: 10.48550/arXiv.1705.07874.
11. A. M. Nawi, R. Ismail, F. Ibrahim, et al., "Risk and protective factors of drug abuse among adolescents: a systematic review," *BMC Public Health*, vol. 21, no. 1, p. 2088, 2021. doi: 10.1186/s12889-021-11906-2.
12. S. Ramírez, S. Gana, M. Godoy, et al., "Validation of the European Drug Addiction Prevention Trial Questionnaire (EU-Dap) for substance use screening and to assess risk and protective factors among early adolescents in Chile," *PLoS ONE*, vol. 16, no. 10, p. e0258288, 2021. doi: 10.1371/journal.pone.0258288.
13. J. Wu, N. Zhang, Q. L. Ye, et al., "DSCnet: detection of drug and alcohol addiction mechanisms based on multi-angle feature learning from the hybrid representation of EEG," *Frontiers in Neuroscience*, vol. 19, p. 1607248, 2025. doi: 10.3389/fnins.2025.1607248.
14. M. M. Habibi, M. Mousavi, M. Shekofteh-Gohari, et al., "Machine learning-enhanced drug testing for simultaneous morphine and methadone detection in urinary biofluids," *Scientific Reports*, vol. 14, p. 58843, 2024. doi: 10.1038/s41598-024-58843-9.
15. S. Li, C.-H. Fan, A. Kargarandehkordi, et al., "Monitoring Substance Use with Fitbit Biosignals: A Case Study on Training Deep Learning Models Using Ecological Momentary Assessments and Passive Sensing," *AI*, vol. 5, no. 4, pp. 1310–1325, 2024. doi: 10.3390/ai5040131.
16. A. Jeyadevan and J. Grigg, "A Time-Limited Scoping Review Investigating the Use of Wearable Biosensors in the Substance Use Field," *Current Addiction Reports*, vol. 11, pp. 587–599, 2024. doi: 10.1007/s40429-024-00587-7.
17. S. Förster, J. A. Callan, J. Cheng, and E. Gordon, "Artificial Intelligence-Enabled, Real-time Risk Monitoring for Substance Use Recovery in Veterans: An Initial Evaluation of the Behavior Recovery Platform," *Military Medicine*, vol. 190, no. Suppl_2, pp. 784–792, 2025. doi: 10.1093/milmed/usaf331.
18. C. Malar, P. E. Chaly, D. Sundareswaran, and A. Ganesh, "Digital droplets: AI in real-time monitoring of substance abuse: a review," *Translational Medicine and Drug Research*, vol. 6, p. 03, 2026. doi: 10.54646/tmdr.2026.03.
19. A. Khakpaki and H. Sepehri, "AI in addiction: Harnessing technology for diagnosis, prevention, and recovery: A narrative review," *Addiction and Substance Abuse*, vol. 3, no. 1, pp. 1–7, 2025. doi: 10.46439/addiction.3.008.
20. V. E. Horigian, R. D. Schmidt, and D. J. Feaster, "Loneliness, Mental Health, and Substance Use among US Young Adults during COVID-19," *Journal of Psychoactive Drugs*, vol. 53, no. 3, pp. 229–240, 2021. doi: 10.1080/02791072.2020.1836435.
21. J. S. Brook, K. Pahl, D. W. Brook, and N. K. Morojele, "Risk and Protective Factors for Substance Use and Abuse," in *Substance Use Disorders in Sub-Saharan Africa*, Springer, 2014, pp. 101–120. doi: 10.1007/978-88-470-5322-9_101.
22. G. W. Joe, W. E. K. Lehman, G. A. Rowan, K. Knight, and P. M. Flynn, "The role of physical and psychological health problems in the drug use treatment process," *Journal of Substance Abuse Treatment*, vol. 102, pp. 11–18, 2019. doi: 10.1016/j.jsat.2019.03.011.
23. O. Wlodarczyk, M. Schwarze, H.-J. Rumpf, F. Metzner, and S. Pawils, "Protective mental health factors in children of parents with alcohol and drug use disorders: a systematic review," *PLoS ONE*, vol. 12, no. 6, p. e0179140, 2017. doi: 10.1371/journal.pone.0179140.
