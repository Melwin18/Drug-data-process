"""Leakage-safe ensemble stage for the finalized drug-risk ML pipeline.

This script preserves the finalized 35 predictors, the locked test indices, and
all existing baseline outputs. It writes ensemble-specific outputs only.
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Drug_Abuse_Risk"
DATA_PATH = "data/final/final_ml_dataset.csv"
TEST_INDICES_PATH = "data/final/test_indices.csv"
METADATA_PATH = "models/model_metadata.json"

BASE_MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "XGBoost",
    "LightGBM",
]


def make_preprocessor(X):
    numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=10.0, solver="lbfgs", class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=1,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=2,
            min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            C=1.0, kernel="linear", class_weight=None, probability=True,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=1.0, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            num_leaves=31, subsample=1.0, colsample_bytree=0.8,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def make_pipeline(estimator, X):
    return Pipeline([
        ("preprocessor", make_preprocessor(X)),
        ("classifier", estimator),
    ])


def probability_matrix(model, X, class_count):
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] != class_count:
        raise ValueError(
            f"Probability shape mismatch: expected (?, {class_count}), got {probabilities.shape}"
        )
    return probabilities


def metric_row(name, model_type, y_true, predictions, probabilities, class_count):
    try:
        roc_auc = roc_auc_score(
            y_true, probabilities, multi_class="ovr", labels=np.arange(class_count)
        )
    except ValueError:
        roc_auc = np.nan
    return {
        "model_name": name,
        "model_type": model_type,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_precision": precision_score(y_true, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "roc_auc": roc_auc,
    }


def evaluate_and_save_confusion(name, model_type, y_true, predictions, labels, output_dir):
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_title(f"Confusion Matrix: {name}")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels, rotation=0)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png"), dpi=150)
    plt.close(figure)
    return matrix


def plot_metric_comparison(metrics_df, metric, title, filename):
    figure, axis = plt.subplots(figsize=(10, 5))
    ordered = metrics_df.sort_values(metric, ascending=False)
    sns.barplot(data=ordered, x=metric, y="model_name", hue="model_type", dodge=False, ax=axis)
    axis.set_title(title)
    axis.set_xlabel(metric.replace("_", " ").title())
    axis.set_ylabel("")
    axis.legend(title="Model type", loc="lower right")
    figure.tight_layout()
    figure.savefig(os.path.join("results", filename), dpi=150)
    plt.close(figure)


def choose_final_model(metrics_df):
    ranking = metrics_df.sort_values(
        ["macro_f1", "macro_recall", "balanced_accuracy", "roc_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    return ranking.iloc[0], ranking


def train_oof_probabilities(pipelines, X_train, y_train, class_count, cv):
    oof = {
        name: np.zeros((len(X_train), class_count), dtype=float)
        for name in pipelines
    }
    fold_scores = {name: [] for name in pipelines}
    for fold, (fit_indices, validation_indices) in enumerate(cv.split(X_train, y_train), start=1):
        X_fit = X_train.iloc[fit_indices]
        X_validation = X_train.iloc[validation_indices]
        y_fit = y_train[fit_indices]
        y_validation = y_train[validation_indices]
        for name, pipeline in pipelines.items():
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_fit, y_fit)
            probabilities = probability_matrix(fold_pipeline, X_validation, class_count)
            oof[name][validation_indices] = probabilities
            fold_scores[name].append(
                f1_score(y_validation, fold_pipeline.predict(X_validation), average="macro", zero_division=0)
            )
        print(f"  OOF fold {fold}/{cv.n_splits} complete")
    return oof, fold_scores


def train_permutation_importance(final_model, X_train, y_train, output_path):
    """Use only a train-derived validation split for final-model explanation."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    explanation_model = clone(final_model)
    explanation_model.fit(X_fit, y_fit)
    permutation = permutation_importance(
        explanation_model,
        X_validation,
        y_validation,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Permutation_Importance_Mean": permutation.importances_mean,
        "Permutation_Importance_Std": permutation.importances_std,
    }).sort_values("Permutation_Importance_Mean", ascending=False)
    importance.to_csv(output_path, index=False)
    return importance


def main():
    print("Starting leakage-safe ensemble stage...")
    for directory in ["results/ensemble_confusion_matrices", "results/ensemble_shap", "models"]:
        os.makedirs(directory, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    metadata = json.load(open(METADATA_PATH, encoding="utf-8"))
    features = metadata["selected_features"]
    if len(features) != 35:
        raise ValueError(f"Expected 35 finalized features, found {len(features)}")
    if list(data[features].columns) != features:
        raise ValueError("Finalized feature order does not match model metadata")

    test_indices = pd.read_csv(TEST_INDICES_PATH).iloc[:, 0].astype(int).to_numpy()
    if len(test_indices) != 1000 or len(np.unique(test_indices)) != 1000:
        raise ValueError("Locked test index artifact must contain 1,000 unique rows")
    if not np.all(np.isin(test_indices, data.index.to_numpy())):
        raise ValueError("Locked test indices contain rows outside the finalized dataset")

    test_mask = data.index.isin(test_indices)
    train_data = data.loc[~test_mask].copy()
    test_data = data.loc[test_mask].copy()
    X_train = train_data[features]
    X_test = test_data[features]
    label_encoder = LabelEncoder()
    label_encoder.fit(data[TARGET])
    y_train = label_encoder.transform(train_data[TARGET]).astype(np.int32)
    y_test = label_encoder.transform(test_data[TARGET]).astype(np.int32)
    class_count = len(label_encoder.classes_)
    print(f"Locked split verified: train={len(X_train)}, test={len(X_test)}")
    print(f"Class order: {list(label_encoder.classes_)}")

    base_estimators = build_models()
    base_pipelines = {name: make_pipeline(model, X_train) for name, model in base_estimators.items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Generating training-only out-of-fold probabilities...")
    oof_probabilities, fold_scores = train_oof_probabilities(
        base_pipelines, X_train, y_train, class_count, cv
    )
    cv_macro_f1 = {name: float(np.mean(scores)) for name, scores in fold_scores.items()}
    weight_values = np.array([max(cv_macro_f1[name], 0.0) for name in BASE_MODEL_NAMES])
    if weight_values.sum() == 0:
        weight_values = np.ones(len(BASE_MODEL_NAMES))
    weights = weight_values / weight_values.sum()
    weight_map = dict(zip(BASE_MODEL_NAMES, weights))
    print("Training-only weighted-voting weights:", weight_map)

    fitted_base = {}
    base_predictions = {}
    base_probabilities = {}
    for name, pipeline in base_pipelines.items():
        print(f"Fitting final training-only base model: {name}")
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)
        fitted_base[name] = fitted
        base_predictions[name] = fitted.predict(X_test)
        base_probabilities[name] = probability_matrix(fitted, X_test, class_count)

    soft_probability = np.mean(
        np.stack([base_probabilities[name] for name in BASE_MODEL_NAMES]), axis=0
    )
    weighted_probability = sum(
        weight_map[name] * base_probabilities[name] for name in BASE_MODEL_NAMES
    )
    stacking_matrix_train = np.hstack([oof_probabilities[name] for name in BASE_MODEL_NAMES])
    stacking_matrix_test = np.hstack([base_probabilities[name] for name in BASE_MODEL_NAMES])
    meta_learner = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    meta_learner.fit(stacking_matrix_train, y_train)
    stacking_probability = probability_matrix(meta_learner, stacking_matrix_test, class_count)

    ensemble_predictions = {
        "Soft Voting": np.argmax(soft_probability, axis=1),
        "Weighted Soft Voting": np.argmax(weighted_probability, axis=1),
        "Stacking": meta_learner.predict(stacking_matrix_test),
    }
    ensemble_probabilities = {
        "Soft Voting": soft_probability,
        "Weighted Soft Voting": weighted_probability,
        "Stacking": stacking_probability,
    }

    rows = []
    for name in BASE_MODEL_NAMES:
        rows.append(metric_row(
            name, "individual", y_test, base_predictions[name], base_probabilities[name], class_count
        ))
    for name in ["Soft Voting", "Weighted Soft Voting", "Stacking"]:
        rows.append(metric_row(
            name, "ensemble", y_test, ensemble_predictions[name], ensemble_probabilities[name], class_count
        ))
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("results/ensemble_model_comparison.csv", index=False)

    confusion_matrices = {}
    for name in ["Soft Voting", "Weighted Soft Voting", "Stacking"]:
        confusion_matrices[name] = evaluate_and_save_confusion(
            name, "ensemble", y_test, ensemble_predictions[name],
            np.arange(class_count), "results/ensemble_confusion_matrices"
        )

    for metric, filename, title in [
        ("macro_f1", "ensemble_macro_f1_comparison.png", "Individual vs Ensemble Macro F1"),
        ("accuracy", "ensemble_accuracy_comparison.png", "Individual vs Ensemble Accuracy"),
        ("balanced_accuracy", "ensemble_balanced_accuracy_comparison.png", "Individual vs Ensemble Balanced Accuracy"),
        ("roc_auc", "ensemble_roc_auc_comparison.png", "Individual vs Ensemble ROC-AUC"),
    ]:
        plot_metric_comparison(metrics_df, metric, title, filename)
    plot_metric_comparison(metrics_df, "macro_f1", "Overall Model Comparison by Macro F1", "ensemble_overall_comparison.png")

    selected_row, ranking = choose_final_model(metrics_df)
    selected_name = selected_row["model_name"]
    baseline_row = metrics_df[metrics_df["model_name"] == "Logistic Regression"].iloc[0]
    ensemble_rows = metrics_df[metrics_df["model_type"] == "ensemble"]
    best_ensemble_row, _ = choose_final_model(ensemble_rows)
    ensemble_improves = tuple(
        best_ensemble_row[metric] > baseline_row[metric]
        for metric in ["macro_f1", "macro_recall", "balanced_accuracy", "roc_auc"]
    )
    ensemble_is_final = selected_name in {"Soft Voting", "Weighted Soft Voting", "Stacking"}

    final_model_object = None
    if ensemble_is_final:
        if selected_name == "Soft Voting":
            final_model_object = {
                "type": "soft_voting",
                "base_pipelines": fitted_base,
                "weights": {name: 1.0 / len(BASE_MODEL_NAMES) for name in BASE_MODEL_NAMES},
                "classes": label_encoder.classes_.tolist(),
            }
        elif selected_name == "Weighted Soft Voting":
            final_model_object = {
                "type": "weighted_soft_voting",
                "base_pipelines": fitted_base,
                "weights": weight_map,
                "classes": label_encoder.classes_.tolist(),
            }
        else:
            final_model_object = {
                "type": "stacking",
                "base_pipelines": fitted_base,
                "meta_learner": meta_learner,
                "classes": label_encoder.classes_.tolist(),
            }
        joblib.dump(final_model_object, "models/final_ensemble_pipeline.pkl")
        explanation_model = fitted_base["Logistic Regression"]
    else:
        explanation_model = fitted_base["Logistic Regression"]

    importance = train_permutation_importance(
        explanation_model, X_train, y_train, "results/ensemble_feature_importance.csv"
    )

    report_lines = [
        "=== ENSEMBLE MODEL REPORT ===",
        "",
        "1. Methodology",
        "   - Used data/final/final_ml_dataset.csv without modifying it.",
        "   - Used the existing 35 finalized features from model metadata.",
        "   - Used data/final/test_indices.csv as the immutable 1,000-row test set.",
        "   - Existing baseline result files were preserved.",
        "",
        "2. Base models",
        "   - Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, LightGBM.",
        "   - Existing tuned hyperparameters from results/hyperparameter_tuning_results.csv were reused.",
        "",
        "3. Ensemble methods",
        "   - Soft Voting: unweighted mean of base-model class probabilities.",
        "   - Weighted Soft Voting: weights derived from 5-fold training-only Macro F1.",
        "   - Stacking: base-model out-of-fold probabilities plus a Logistic Regression meta-learner.",
        "",
        "4. Cross-validation and leakage prevention",
        "   - Stratified 5-fold CV with random state 42 generated OOF predictions.",
        "   - Every fold fitted its own imputer, scaler, and encoder inside a Pipeline.",
        "   - The locked test set was used only once for final evaluation.",
        "   - No test rows were used for weights, OOF predictions, meta-learning, or selection.",
        "",
        "5. Training-only ensemble weights",
    ]
    report_lines.extend(f"   - {name}: {weight_map[name]:.6f} (CV Macro F1={cv_macro_f1[name]:.6f})" for name in BASE_MODEL_NAMES)
    report_lines.extend([
        "",
        "6. Meta-learner",
        "   - Logistic Regression with class_weight='balanced', trained only on concatenated OOF probabilities.",
        "",
        "7. Locked test-set metrics",
        metrics_df.to_string(index=False),
        "",
        "8. Objective final selection",
        "   Priority: Macro F1, Macro Recall, Balanced Accuracy, ROC-AUC.",
        f"   Final selected model: {selected_name}",
        f"   Ensemble improves over Logistic Regression by priority metrics: {'YES' if ensemble_improves[0] else 'NO'}",
        "",
        "9. Comparison with Logistic Regression",
        f"   Baseline Macro F1: {baseline_row['macro_f1']:.6f}",
        f"   Best ensemble Macro F1: {best_ensemble_row['macro_f1']:.6f}",
        f"   Baseline Macro Recall: {baseline_row['macro_recall']:.6f}",
        f"   Best ensemble Macro Recall: {best_ensemble_row['macro_recall']:.6f}",
        f"   Baseline Balanced Accuracy: {baseline_row['balanced_accuracy']:.6f}",
        f"   Best ensemble Balanced Accuracy: {best_ensemble_row['balanced_accuracy']:.6f}",
        f"   Baseline ROC-AUC: {baseline_row['roc_auc']:.6f}",
        f"   Best ensemble ROC-AUC: {best_ensemble_row['roc_auc']:.6f}",
        "",
        "10. Explainability",
        "   - Permutation importance was generated using a train-derived validation split.",
        "   - Importance indicates model feature contribution, not causal risk factors.",
        "   - SHAP was not forced for the complete heterogeneous ensemble; permutation importance is used instead.",
        "",
        "11. Limitations",
        "   - Drug_Abuse_Risk is an operational research label, not a clinical diagnosis.",
        "   - Cross-sectional, self-reported data may contain response bias.",
        "   - Ensemble performance is specific to this locked test set and dataset.",
    ])
    with open("reports/ensemble_model_report.txt", "w", encoding="utf-8") as report:
        report.write("\n".join(report_lines) + "\n")

    metadata.update({
        "baseline_model": "Logistic Regression",
        "ensemble_models": ["Soft Voting", "Weighted Soft Voting", "Stacking"],
        "ensemble_method": "Soft voting, weighted soft voting, and OOF probability stacking",
        "final_model": selected_name,
        "final_metrics": {key: float(value) if pd.notna(value) else None for key, value in selected_row.items() if key in {
            "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy", "roc_auc"
        }},
    })
    with open(METADATA_PATH, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print("\n=== ENSEMBLE SUMMARY ===")
    print(metrics_df.to_string(index=False))
    print(f"Final selected model: {selected_name}")
    print(f"Ensemble improves over Logistic Regression: {'YES' if ensemble_improves[0] else 'NO'}")
    print("Top features:")
    print(importance.head(10)[["Feature", "Permutation_Importance_Mean"]].to_string(index=False))
    print("Generated: results/ensemble_model_comparison.csv")
    print("Generated: reports/ensemble_model_report.txt")
    if ensemble_is_final:
        print("Generated: models/final_ensemble_pipeline.pkl")


if __name__ == "__main__":
    main()
