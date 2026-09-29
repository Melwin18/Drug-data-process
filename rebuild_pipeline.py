"""
rebuild_pipeline.py
-------------------
Rebuilds best_model_pipeline.pkl from scratch using the CURRENT
scikit-learn version, so there are no _fill_dtype or serialisation
compatibility issues.

This does NOT retrain the classifier — it:
  1. Fits a fresh preprocessor on the training split of final_ml_dataset.csv
  2. Wraps the SAME LogisticRegression coefficients/intercept into a new LR object
  3. Saves a clean Pipeline that works with the installed sklearn version

Run once:  python rebuild_pipeline.py
"""

import json, joblib, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# ── Load data ──────────────────────────────────────────────────────────────
df   = pd.read_csv("data/final/final_ml_dataset.csv")
meta = json.load(open("models/model_metadata.json"))

TARGET   = "Drug_Abuse_Risk"
features = meta["selected_features"]

X = df[features]
y_raw = df[TARGET]

# Encode target the same way training did (alphabetical LabelEncoder)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Class mapping:", dict(zip(le.transform(le.classes_), le.classes_)))
print("Expected order:", meta["target_classes"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ── Define column groups (same as original training) ──────────────────────
num_cols = [
    "Financial_Stress", "Hopelessness_Level", "Social_Isolation",
    "Childhood_Adversity", "Sleep_Quality", "Workload_Level",
    "Behavioural_Problems", "Night_Time_Device_Usage", "Depression_Score",
    "Stress_Level", "Loneliness_Level", "Impulsivity", "Psychological_Distress",
    "Self_Esteem", "Screen_Time", "Anxiety_Level", "Peer_Substance_Exposure",
    "Sleep_Duration", "Family_Conflict", "Social_Support", "Academic_Work_Stress",
    "Social_Pressure", "Substance_Use_Intention", "Academic_Performance",
    "Family_Stability", "Emotional_Regulation", "Family_Support", "Self_Control",
    "Job_Satisfaction", "Peer_Influence", "Decision_Making_Control",
    "Awareness_of_Drug_Related_Risks", "School_Work_Absenteeism",
]
cat_cols = ["Previous_Substance_Exposure", "Attitude_Toward_Substance_Use"]

# Use the EXACT same categories the original encoder saw so the output
# shape matches the classifier's expected 40 features.
known_categories = [
    np.array(["No", "Yes"], dtype=object),                                                 # Previous_Substance_Exposure
    np.array(["Accepting", "Approving", "Disapproving", "Neutral", "Strongly disapproving"], dtype=object),  # Attitude
]

# ── Build fresh preprocessor and fit on training data ─────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(categories=known_categories, drop=None, sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])

# Fit preprocessor on training split only
preprocessor.fit(X_train)
print("Preprocessor fitted on", X_train.shape[0], "rows")

# ── Load existing classifier and copy its learned weights ─────────────────
old_pipe = joblib.load("models/best_model_pipeline.pkl")
old_clf  = old_pipe.named_steps["classifier"]

new_clf = LogisticRegression(
    C=10.0,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE,
)
# Transfer the learned coefficients — no retraining
new_clf.coef_      = old_clf.coef_.copy()
new_clf.intercept_ = old_clf.intercept_.copy()
new_clf.classes_   = old_clf.classes_.copy()
new_clf.n_features_in_ = old_clf.n_features_in_
print("Classifier coefficients transferred")

# ── Assemble and save ─────────────────────────────────────────────────────
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   new_clf),
])

# Quick validation
X_sample = X_test.head(5)
pred  = full_pipeline.predict(X_sample)
proba = full_pipeline.predict_proba(X_sample)
print("Sample predictions:", pred)
print("Sample probas:     ", proba.round(3))

joblib.dump(full_pipeline, "models/best_model_pipeline.pkl")
print("Saved models/best_model_pipeline.pkl")
print("Done.")
