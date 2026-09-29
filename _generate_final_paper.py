from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

metrics = pd.read_csv(RESULTS / "ensemble_model_comparison.csv")
metrics = metrics.rename(columns={"model_name": "Model", "accuracy": "Accuracy", "macro_precision": "Macro Precision", "macro_recall": "Macro Recall", "macro_f1": "Macro F1", "balanced_accuracy": "Balanced Accuracy", "roc_auc": "ROC-AUC"})
order = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "XGBoost", "LightGBM", "Soft Voting", "Weighted Soft Voting", "Stacking"]
metrics["sort"] = metrics["Model"].map({name: i for i, name in enumerate(order)})
metrics = metrics.sort_values("sort").drop(columns=["sort", "model_type"], errors="ignore")
metrics.to_csv(RESULTS / "final_model_comparison_updated.csv", index=False)

for metric, filename, title in [("Macro F1", "final_macro_f1_comparison.png", "Locked-test Macro F1 comparison"), ("Accuracy", "final_accuracy_comparison.png", "Locked-test Accuracy comparison"), ("ROC-AUC", "final_roc_auc_comparison.png", "Locked-test ROC-AUC comparison")]:
    plot = metrics.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#0f766e" if x == "Logistic Regression" else ("#f59e0b" if x in {"Soft Voting", "Weighted Soft Voting", "Stacking"} else "#64748b") for x in plot["Model"]]
    ax.barh(plot["Model"], plot[metric], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=.25)
    for i, value in enumerate(plot[metric]): ax.text(value + .008, i, f"{value:.4f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / filename, dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 3.4))
steps = ["Raw dataset", "Audit & cleaning", "Leakage assessment", "35 predictors", "Locked split", "Preprocessing", "Six models", "Ensembles", "Selection", "SHAP", "Streamlit"]
ax.plot(range(len(steps)), [0]*len(steps), alpha=0)
for i, step in enumerate(steps):
    ax.text(i, 0, step, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=.55", fc="#e6fffb" if i in {3,8,9} else "#f1f5f9", ec="#0f766e" if i in {3,8,9} else "#94a3b8"))
    if i < len(steps)-1: ax.annotate("", xy=(i+.82, 0), xytext=(i+.22, 0), arrowprops=dict(arrowstyle="->", color="#475569"))
ax.set_xlim(-.6, len(steps)-.4); ax.set_ylim(-.8, .8); ax.axis("off"); fig.tight_layout(); fig.savefig(RESULTS / "final_methodology_architecture.png", dpi=180); plt.close(fig)

features = [line.strip(" -\n") for line in (REPORTS / "final_feature_set.txt").read_text(encoding="utf-8", errors="replace").splitlines() if line.strip() and not line.lower().startswith(("final", "selected", "number", "features:"))]
features_text = ", ".join(features) if features else "The 35 predictors listed in reports/final_feature_set.txt."

def fmt(x): return f"{x:.6f}"

def md_table(df):
    cols = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Balanced Accuracy", "ROC-AUC"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows(): lines.append("| " + " | ".join([str(r["Model"]), f'{r["Accuracy"]:.3f}', fmt(r["Macro Precision"]), fmt(r["Macro Recall"]), fmt(r["Macro F1"]), fmt(r["Balanced Accuracy"]), fmt(r["ROC-AUC"])]) + " |")
    return "\n".join(lines)

table = md_table(metrics)
refs = """1. F. Bulut and İ. Ö. Bucak, \"An urgent precaution system to detect students at risk of substance abuse through classification algorithms,\" *Turkish Journal of Electrical Engineering & Computer Sciences*, vol. 22, no. 5, pp. 1208–1220, 2014. doi: 10.3906/elk-1208-60.
2. P. Nath, S. Kilam, and A. Swetapadma, \"A machine learning approach to predict volatile substance abuse for drug risk analysis,\" in *2017 2nd International Conference on Recent Trends in Electronics, Information & Communication Technology*, 2017, pp. 823–827. doi: 10.1109/ICRCICN.2017.8234516.
3. Z. Qiao, T. Chai, Q. Zhang, X. Zhou, and Z. Chu, \"Predicting potential drug abusers using machine learning techniques,\" in *2019 IEEE 4th International Conference on Cloud Computing and Big Data Analytics*, 2019, pp. 490–494. doi: 10.1109/ICIIBMS46890.2019.8991550.
4. Y. Jing et al., \"Analysis of substance use and its outcomes by machine learning I. Childhood Evaluation of Liability to Substance Use Disorder,\" *Drug and Alcohol Dependence*, vol. 204, p. 107605, 2019. doi: 10.1016/j.drugalcdep.2019.107605.
5. D.-H. Han, S. Lee, and D. Seo, \"Using machine learning to predict opioid misuse among U.S. adolescents,\" *Preventive Medicine*, vol. 130, p. 105886, 2020. doi: 10.1016/j.ypmed.2019.105886.
6. A. S. Wadekar, \"A Psychosocial Approach to Predicting Substance Use Disorder Among Adolescents,\" in *2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining*, 2020, pp. 784–788. doi: 10.1109/ASONAM49781.2020.9381378.
7. U. I. Islam et al., \"Predicting Individual Substance Abuse Vulnerability using Machine Learning Techniques,\" in *Security, Privacy, and Anonymity in Computation, Communication, and Storage*, 2021, pp. 505–516. doi: 10.1007/978-3-030-73050-5_42.
8. S. Kim et al., \"Machine Learning-Based Prediction of Substance Use in Adolescents in Three Independent Worldwide Cohorts,\" *Journal of Medical Internet Research*, vol. 26, p. e62805, 2024. doi: 10.2196/62805.
9. A. Ünlü and A. Subaşı, \"Substance use prediction using artificial intelligence techniques,\" *Journal of Computational Social Science*, vol. 7, pp. 356–375, 2024. doi: 10.1007/s42001-024-00356-6.
10. S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in *Advances in Neural Information Processing Systems*, vol. 30, 2017. doi: 10.48550/arXiv.1705.07874.
11. A. M. Nawi et al., \"Risk and protective factors of drug abuse among adolescents: a systematic review,\" *BMC Public Health*, vol. 21, p. 2088, 2021. doi: 10.1186/s12889-021-11906-2.
12. S. Ramírez et al., \"Validation of the European Drug Addiction Prevention Trial Questionnaire for substance use screening and risk and protective factors,\" *PLoS ONE*, vol. 16, p. e0258288, 2021. doi: 10.1371/journal.pone.0258288."""

paper = f'''# An Explainable Machine Learning Framework for Early Drug-Abuse Risk Assessment and Stratification

**Authors and affiliations:** To be completed by the authors.

## Abstract
Substance-use risk assessment involves psychological, behavioural, social, familial, lifestyle, academic/work, and exposure-related information that is difficult to represent with a single-factor model. This study audits and evaluates an explainable multiclass framework using 35 finalized predictors to assign assessed individuals to Low Risk, Moderate Risk, or High Risk. Six individual classifiers—Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, XGBoost, and LightGBM—were evaluated on a locked 1,000-row test set after development on 4,000 observations. Soft voting, weighted soft voting, and stacking were additionally investigated to determine whether combining heterogeneous classifiers improved stratification. Logistic Regression achieved the highest performance in the present evaluation: 70.5% accuracy, approximately 70.0% Macro F1, and ROC-AUC 0.863064. None of the tested ensembles exceeded this baseline. SHAP analysis was retained for model-associated feature attribution. The system is an early risk-assessment and decision-support application, not a clinical diagnostic system or a validated model for first-time substance use, future addiction, or other prospective outcomes.

**Keywords:** substance-use risk assessment; multiclass classification; explainable machine learning; ensemble learning; SHAP; decision support.

## 1. INTRODUCTION
### 1.1 Background
Machine learning has been investigated for substance-use risk assessment and vulnerability classification [1–9]. The relevant signals span mental health, social context, family environment, behaviour, lifestyle, and prior exposure [11, 12]. This motivates an integrative framework that supports structured assessment while preserving a clear distinction between model output and clinical diagnosis.

### 1.2 Problem Statement
The project classifies assessed records into three operational categories: Low Risk, Moderate Risk, and High Risk. The available artifacts do not establish that these labels are clinically validated diagnoses or prospective outcomes. Their construction therefore requires author confirmation.

### 1.3 Research Gap
A practical gap remains between comparative multiclass modelling, leakage-aware feature selection, explanation, and application delivery in one auditable workflow. Ensemble learning was added as an experiment because combining heterogeneous classifiers may improve multiclass risk stratification, but improvement cannot be assumed.

### 1.4 Objectives
The objectives were to preserve the finalized 35 predictors and locked test set, compare six individual classifiers, evaluate voting and stacking ensembles, select the strongest model under the project's evaluation criteria, and provide model-associated explanations through SHAP [10].

### 1.5 Contributions
The framework contributes: (1) multidimensional risk-factor integration; (2) leakage-aware feature selection; (3) comparative evaluation of six heterogeneous classifiers; (4) an ensemble-learning investigation using voting and stacking; (5) SHAP-based explanation; (6) three-level risk stratification; and (7) an application-oriented Streamlit implementation. No individual algorithm or explanation method is claimed as novel.

## 2. RELATED WORK
### 2.1 Machine Learning for Substance-Use Risk Assessment
Prior studies have examined classification and prediction for student substance-use risk, potential drug abuse, opioid misuse, and vulnerability [1–9]. Their targets, populations, and validation settings differ, so reported performance should not be interpreted as directly comparable to this project.

### 2.2 Multidimensional Risk Factors
Reviews identify interacting psychological, family, peer, social, and behavioural risk and protective factors [11, 12]. The present framework uses these domains as predictors for operational stratification; it does not estimate causal effects.

### 2.3 Explainable AI
SHAP provides feature-attribution values for model predictions [10]. In this study, SHAP is interpreted as model feature attribution, not as evidence that a feature causes drug abuse.

### 2.4 Research Gap
The project addresses an applied integration gap by joining audit, feature selection, locked evaluation, ensemble comparison, explanation, and Streamlit deployment. External validation remains necessary.

## 3. METHODOLOGY
### 3.1 Study Design
The workflow was: raw dataset → data audit → cleaning → leakage assessment → feature selection → 35 predictors → locked 80:20 split → preprocessing → six individual models → ensemble learning → model comparison → final model selection → SHAP → Streamlit application. The system is intended for decision support and early risk assessment.

### 3.2 Dataset
The finalized dataset contains 5,000 observations, 87 raw columns, 35 finalized predictors, and the target `Drug_Abuse_Risk`. The observed class totals are Low Risk = 2,000, Moderate Risk = 1,500, and High Risk = 1,500. The project artifacts do not establish the dataset's source, whether it is public, private, synthetic, generated, or constructed, whether records were combined from multiple datasets, or what population they represent. **[DATASET PROVENANCE REQUIRES AUTHOR CONFIRMATION]**

The final predictor set was preserved from `reports/final_feature_set.txt`: {features_text}

### 3.3 Data Quality and Leakage Assessment
The project generated data-quality, correlation, redundancy, feature-selection, and leakage-audit reports. Features identified as leakage-prone or unsuitable were excluded before final modelling. The finalized 35 predictors were not changed for the ensemble experiment.

### 3.4 Feature Selection
Feature selection combined the generated feature-selection and leakage-audit artifacts. The resulting 35 predictors were used consistently by the individual and ensemble pipelines. The selection process should be read as a project-specific preprocessing and audit decision, not as evidence of causal importance.

### 3.5 Data Preprocessing
The implementation uses model pipelines so preprocessing is fitted on training data. The ensemble code explicitly clones and fits a separate pipeline in each cross-validation fold, preventing fold validation data from fitting imputation, scaling, or encoding transformations.

### 3.6 Train-Test Split
`data/final/test_indices.csv` defines 1,000 unique locked test rows. The remaining 4,000 observations form the development training set. The locked test set is reserved for final evaluation and was not used for ensemble weights, out-of-fold predictions, meta-learning, or model selection.

### 3.7 Individual Machine Learning Models
The evaluated individual models were Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, and LightGBM. The implementation preserves the generated model configurations and reports; this paper does not replace them with theoretical or undocumented hyperparameters.

### 3.8 Ensemble Learning
Ensemble learning was introduced **to investigate whether combining heterogeneous classifiers could improve multiclass risk stratification compared with individual classifiers**. Soft Voting averages the base-model class probabilities. Weighted Soft Voting uses model weights derived from training-only five-fold cross-validation Macro F1 values. Stacking constructs out-of-fold base-model probability features using five-fold stratified cross-validation and trains a Logistic Regression meta-learner on those training-only features. Each fold fits its own preprocessing pipeline. The locked test set is used only to generate final evaluation probabilities and predictions.

### 3.9 Evaluation Metrics
Accuracy, macro precision, macro recall, macro F1, balanced accuracy, and multiclass ROC-AUC were reported. Macro metrics give each class equal weight, while balanced accuracy summarizes class-wise recall.

### 3.10 SHAP Explainability
SHAP was retained for the final Logistic Regression model because Logistic Regression remained selected. The documented influential features were Previous Substance Exposure, Peer Substance Exposure, Depression Score, Stress Level, Impulsivity, Psychological Distress, Anxiety Level, Peer Influence, Family Conflict, and Social Isolation. These are model-associated contributions; SHAP does not establish causal effects.

### 3.11 Streamlit Deployment
The Streamlit application provides the application workflow for risk assessment, model performance, feature importance, model information, and related project outputs. It is an application interface for decision support, not a clinical diagnostic service.

**Figure 1.** Audited project workflow from raw data to Streamlit deployment.  
![Audited methodology architecture](results/final_methodology_architecture.png)

## 4. RESULTS
### 4.1 Individual Model Performance
Table 1 reports all six individual models on the locked test set. Logistic Regression had the highest Accuracy, Macro F1, Balanced Accuracy, and ROC-AUC in this evaluation.

**Table 1. Individual and ensemble model comparison on the locked 1,000-row test set.**

{table}

### 4.2 Ensemble Model Performance
Soft Voting reached Accuracy 0.685 and Macro F1 {metrics.loc[metrics.Model == 'Soft Voting', 'Macro F1'].iloc[0]:.6f}. Weighted Soft Voting reached Accuracy 0.687 and Macro F1 {metrics.loc[metrics.Model == 'Weighted Soft Voting', 'Macro F1'].iloc[0]:.6f}. Stacking was the strongest ensemble on several metrics, with Accuracy 0.693, Macro F1 {metrics.loc[metrics.Model == 'Stacking', 'Macro F1'].iloc[0]:.6f}, and ROC-AUC {metrics.loc[metrics.Model == 'Stacking', 'ROC-AUC'].iloc[0]:.6f}; it nevertheless remained below Logistic Regression.

### 4.3 Individual vs Ensemble Comparison
Figures 2–4 show the complete comparisons. Logistic Regression remains above all three ensemble configurations for Accuracy, Macro F1, Balanced Accuracy, and ROC-AUC. The result is specific to this dataset, predictor representation, split, and implementation.

**Figure 2.** Complete locked-test Macro F1 comparison.  
![Macro F1 comparison](results/final_macro_f1_comparison.png)

**Figure 3.** Complete locked-test Accuracy comparison.  
![Accuracy comparison](results/final_accuracy_comparison.png)

**Figure 4.** Complete locked-test ROC-AUC comparison.  
![ROC-AUC comparison](results/final_roc_auc_comparison.png)

### 4.4 Confusion Matrix Analysis
The project generated individual confusion matrices and ensemble confusion matrices in `results/confusion_matrices/` and `results/ensemble_confusion_matrices/`. These matrices should be read alongside macro recall and balanced accuracy; no class-level causal or clinical conclusion is warranted from them.

### 4.5 Final Model Selection
Logistic Regression is the final selected model because it achieved the strongest performance under the project's model-selection criteria. **Although ensemble learning was evaluated as an enhancement, none of the tested ensemble configurations exceeded the Logistic Regression baseline on the locked test set.** This result indicates that greater model complexity does not necessarily guarantee improved performance for this dataset and predictor representation; it does not establish universal superiority of Logistic Regression.

### 4.6 SHAP-Based Feature Contributions
The retained SHAP analysis identifies the ten features listed in Section 3.10 as influential in the fitted Logistic Regression model. These rankings describe how features contributed to model outputs in the evaluated data. They are not causal estimates and should not be interpreted as claims that the features cause drug abuse.

### 4.7 Application Workflow
The Streamlit application exposes assessment, performance, feature-importance, and model-information views. Its outputs should be interpreted by an appropriately responsible human user and not treated as an autonomous diagnosis or guaranteed intervention recommendation.

## 5. DISCUSSION
The individual comparison shows that the linear Logistic Regression model was strongest in this experimental setting, while SVM was the closest individual alternative by Macro F1 and ROC-AUC. The ensemble experiment did not improve the baseline: Stacking was the closest ensemble, but its locked-test metrics remained lower. Possible explanations include correlated errors among base models, the information content and representation of the 35 predictors, and the fact that probability aggregation cannot automatically correct systematic class-boundary errors. These are interpretations of the observed comparison, not experimentally established causes.

The multidimensional feature design is appropriate for representing risk assessment as a structured combination of domains, and the three-level output supports more differentiated stratification than a binary decision. However, the operational meaning and construction of the labels require author confirmation. SHAP improves transparency by showing model-associated feature contributions, while avoiding a causal interpretation. The Streamlit implementation demonstrates practical delivery, but deployment does not replace external, prospective, fairness, calibration, or clinical validation.

## 6. LIMITATIONS
The dataset provenance, population, record-construction process, and target-label construction are not sufficiently documented in the available project artifacts and require author confirmation. The evaluation uses one locked test set from the finalized dataset, so generalization is unknown. No prospective validation, external cohort validation, calibration study, subgroup fairness analysis, or clinical validation is established by the artifacts reviewed. The target is an operational risk classification, not a validated diagnosis and not a validated prediction of first-time substance use or future addiction. Observational feature associations and SHAP attributions must not be interpreted causally. The application is decision support and should remain subject to human oversight.

## 7. CONCLUSION AND FUTURE WORK
This audit confirms an explainable, multidimensional, 35-predictor framework with six individual classifiers and an additional voting/stacking experiment. On the locked 1,000-row test set, Logistic Regression remained the final model with Accuracy 0.705, Macro F1 0.700119, Balanced Accuracy 0.699444, and ROC-AUC 0.863064. None of the tested ensembles improved those results. Future work should document dataset provenance and label construction, validate the system externally and prospectively, assess calibration and subgroup performance, and evaluate whether alternative representations or ensemble designs improve robustness without weakening interpretability.

## REFERENCES
{refs}
'''
(REPORTS / "updated_research_paper.md").write_text(paper, encoding="utf-8")

# Build DOCX from the revised Markdown-like structure.
doc = Document(); sec = doc.sections[0]; sec.top_margin = Inches(.7); sec.bottom_margin = Inches(.7); sec.left_margin = Inches(.8); sec.right_margin = Inches(.8)
styles = doc.styles; styles['Normal'].font.name = 'Aptos'; styles['Normal'].font.size = Pt(10)
for line in paper.splitlines():
    if not line.strip(): continue
    if line.startswith('# ') or line.startswith('## '):
        level = 0 if line.startswith('# ') else 1; p = doc.add_heading(line.lstrip('# ').strip(), level=level)
    elif line.startswith('### '): doc.add_heading(line[4:].strip(), level=2)
    elif line.startswith('**Figure') or line.startswith('**Table'):
        p = doc.add_paragraph(); p.add_run(line.replace('**','')).bold = True
    elif line.startswith('|'):
        continue
    elif line.startswith('![Audit') or line.startswith('![Macro') or line.startswith('![Accuracy') or line.startswith('![ROC'):
        img = line.split('](')[-1].rstrip(')'); doc.add_picture(str(ROOT / img), width=Inches(6.4)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif line.startswith('**Authors') or line.startswith('**Keywords'):
        doc.add_paragraph(line.replace('**',''))
    else:
        doc.add_paragraph(line.replace('`',''))
# Insert a clean comparison table at the end of the results text would require parsing; append labeled table as an explicit appendix table.
doc.add_heading('Table 1. Complete locked-test model comparison', level=2)
t = doc.add_table(rows=1, cols=7); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.style = 'Table Grid'
for i, col in enumerate(["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Balanced Accuracy", "ROC-AUC"]): t.rows[0].cells[i].text = col
for _, r in metrics.iterrows():
    cells = t.add_row().cells
    vals = [r['Model'], f"{r['Accuracy']:.3f}", fmt(r['Macro Precision']), fmt(r['Macro Recall']), fmt(r['Macro F1']), fmt(r['Balanced Accuracy']), fmt(r['ROC-AUC'])]
    for i, val in enumerate(vals): cells[i].text = str(val); cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

doc.save(ROOT / "FINAL_RESEARCH_PAPER_ENSEMBLE_ENHANCED.docx")

baseline = metrics[metrics.Model == 'Logistic Regression'].iloc[0]
ensembles = metrics[metrics.Model.isin(["Soft Voting", "Weighted Soft Voting", "Stacking"])]
audit = f'''FINAL RESEARCH PAPER AND PROJECT AUDIT\n=====================================\n\nWHAT WAS CHANGED\n- Rewrote reports/updated_research_paper.md into the requested seven-part research-paper structure.\n- Created FINAL_RESEARCH_PAPER_ENSEMBLE_ENHANCED.docx.\n- Created results/final_model_comparison_updated.csv with all six individual and three ensemble models.\n- Created refreshed comparison figures: final_macro_f1_comparison.png, final_accuracy_comparison.png, final_roc_auc_comparison.png.\n- Created final_methodology_architecture.png.\n- Preserved the existing SHAP interpretation for Logistic Regression and documented model attribution versus causality.\n\nWHAT WAS VERIFIED\n- Final dataset: 5,000 rows, 87 raw columns, 35 finalized predictors.\n- Class totals: Low Risk 2,000; Moderate Risk 1,500; High Risk 1,500.\n- Locked split: 4,000 development rows and 1,000 test rows from data/final/test_indices.csv.\n- Individual models: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, LightGBM.\n- Ensembles: Soft Voting, Weighted Soft Voting, Stacking.\n- Ensemble implementation: five-fold stratified CV; fold-local preprocessing; training-only CV-derived weighted-voting weights; OOF probabilities for stacking; Logistic Regression meta-learner; locked test set reserved for final evaluation.\n- Final model: Logistic Regression.\n- SHAP influential features documented by the project: Previous Substance Exposure; Peer Substance Exposure; Depression Score; Stress Level; Impulsivity; Psychological Distress; Anxiety Level; Peer Influence; Family Conflict; Social Isolation.\n- Existing bibliography entries and DOI information were preserved for the references cited in the revised paper.\n\nFINAL MODEL AND BASELINE METRICS\n- Logistic Regression: Accuracy={baseline['Accuracy']:.6f}; Macro Precision={baseline['Macro Precision']:.6f}; Macro Recall={baseline['Macro Recall']:.6f}; Macro F1={baseline['Macro F1']:.6f}; Balanced Accuracy={baseline['Balanced Accuracy']:.6f}; ROC-AUC={baseline['ROC-AUC']:.6f}.\n\nENSEMBLE METRICS\n'''
for _, r in ensembles.iterrows(): audit += f"- {r['Model']}: Accuracy={r['Accuracy']:.6f}; Macro Precision={r['Macro Precision']:.6f}; Macro Recall={r['Macro Recall']:.6f}; Macro F1={r['Macro F1']:.6f}; Balanced Accuracy={r['Balanced Accuracy']:.6f}; ROC-AUC={r['ROC-AUC']:.6f}.\n"
audit += '''\nWHAT COULD NOT BE VERIFIED\n- [DATASET PROVENANCE REQUIRES AUTHOR CONFIRMATION]: the artifacts do not establish source, public/private status, synthetic/generated/constructed status, dataset combination history, or represented population.\n- [TARGET LABEL CONSTRUCTION REQUIRES AUTHOR CONFIRMATION]: the artifacts do not sufficiently document how Low Risk, Moderate Risk, and High Risk labels were created or whether they are dataset-provided, rule-based, generated, or questionnaire-derived.\n\nFINAL CONSISTENCY CONCLUSION\n- Logistic Regression remains final; no ensemble is described as final and no ensemble improvement is claimed.\n- The paper states that the target is operational risk classification, not clinical diagnosis, causal effect, prospective first-use prediction, or future addiction prediction.\n- Remaining limitations include single-dataset/locked-test evaluation, absent external or prospective validation, unknown calibration and subgroup fairness, and unresolved provenance and label-construction documentation.\n'''
(REPORTS / "FINAL_PAPER_AUDIT_REPORT.txt").write_text(audit, encoding="utf-8")
print('Generated final paper, DOCX, audit report, updated table, and figures.')
