import os
import json
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file, works regardless of CWD
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))   # app/utils/
_APP_DIR = os.path.dirname(_HERE)                        # app/
BASE_DIR = os.path.dirname(_APP_DIR)                     # project root

MODELS_DIR         = os.path.join(BASE_DIR, "models")
RESULTS_DIR        = os.path.join(BASE_DIR, "results")
FEATURE_SPECS_FILE = os.path.join(BASE_DIR, "feature_specs.json")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the full sklearn Pipeline (preprocessor + classifier).
    Raises FileNotFoundError / RuntimeError with a clear message on failure.
    Returns (pipeline, None) on success or (None, error_message) on failure.
    """
    path = os.path.join(MODELS_DIR, "best_model_pipeline.pkl")
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, f"Failed to deserialise model pipeline: {e}"


@st.cache_data(show_spinner=False)
def load_metadata():
    """Load model_metadata.json. Returns dict; contains 'error' key on failure."""
    path = os.path.join(MODELS_DIR, "model_metadata.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def load_feature_specs():
    """Load feature_specs.json. Returns dict; contains 'error' key on failure."""
    try:
        with open(FEATURE_SPECS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Metadata validation helper
# ---------------------------------------------------------------------------
REQUIRED_METADATA_KEYS = ["selected_features", "target_classes", "model_name"]

def validate_metadata(metadata: dict) -> list:
    """Return list of missing required keys."""
    return [k for k in REQUIRED_METADATA_KEYS if k not in metadata]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def create_input_dataframe(user_inputs: dict, expected_features: list) -> pd.DataFrame:
    """
    Create a 1-row DataFrame with columns ordered exactly as during training.
    The pipeline's ColumnTransformer selects columns by name, so this
    enforces the training column order.
    """
    df = pd.DataFrame([user_inputs])
    df = df[expected_features]
    return df


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def get_prediction_and_proba(model, df_input: pd.DataFrame, target_classes: list):
    """
    Run prediction and probability estimation.

    Returns:
        pred_class (str): one of the target_classes strings
        proba (list[float] | None): probabilities in target_classes order
        prob_classes (list[str]): class labels corresponding to proba values
    """
    raw_pred = model.predict(df_input)[0]

    # predict() may return an integer index or a string label
    if isinstance(raw_pred, str):
        pred_class = raw_pred
    else:
        pred_class = target_classes[int(raw_pred)]

    has_proba   = hasattr(model, "predict_proba")
    proba       = None
    prob_classes = []

    if has_proba:
        try:
            proba = model.predict_proba(df_input)[0]
        except (AttributeError, RuntimeError, ValueError):
            # Older serialized sklearn estimators may not support probabilities
            # after being loaded by a newer or older sklearn runtime.
            proba = None

        # Resolve the class order that predict_proba follows (model.classes_)
        clf = model.named_steps.get("classifier", None) if hasattr(model, "named_steps") else model
        if proba is not None and clf is not None and hasattr(clf, "classes_"):
            raw_classes = list(clf.classes_)
            # Map integer class indices → string labels from target_classes
            if raw_classes and not isinstance(raw_classes[0], str):
                try:
                    prob_classes = [target_classes[int(c)] for c in raw_classes]
                except (IndexError, ValueError):
                    prob_classes = [str(c) for c in raw_classes]
            else:
                prob_classes = [str(c) for c in raw_classes]
        else:
            prob_classes = target_classes  # fallback

    return pred_class, proba, prob_classes


# ---------------------------------------------------------------------------
# Recommendations (non-diagnostic, general)
# ---------------------------------------------------------------------------
def generate_recommendations(risk_level: str) -> list:
    """Return general, non-diagnostic preventive recommendations."""
    if risk_level == "Low Risk":
        return [
            "Maintain healthy sleep habits and a consistent daily routine.",
            "Keep up positive social and family support networks.",
            "Continue practising healthy stress-management techniques (e.g. exercise, hobbies).",
        ]
    elif risk_level == "Moderate Risk":
        return [
            "Consider speaking with a counsellor or trusted professional about current stressors.",
            "Strengthen family and social support connections.",
            "Address persistent emotional difficulties proactively before they escalate.",
            "Review lifestyle habits — sleep, screen time, and physical activity.",
        ]
    elif risk_level == "High Risk":
        return [
            "Encourage consultation with a qualified healthcare or mental-health professional promptly.",
            "Explore appropriate counselling or evidence-based support programmes.",
            "If someone is in immediate danger, contact appropriate emergency or crisis services.",
            "Seek peer support programmes and community resources where available.",
        ]
    return []


# ---------------------------------------------------------------------------
# Domain / feature grouping
# ---------------------------------------------------------------------------
def get_feature_domains() -> dict:
    """Maps the 35 features to their respective domains for UI organisation."""
    return {
        "Demographic": [
            "Financial_Stress",
        ],
        "Psychological": [
            "Hopelessness_Level", "Depression_Score", "Stress_Level",
            "Loneliness_Level", "Impulsivity", "Psychological_Distress",
            "Self_Esteem", "Anxiety_Level", "Emotional_Regulation",
        ],
        "Behavioural": [
            "Behavioural_Problems", "Self_Control", "Decision_Making_Control",
        ],
        "Social / Peer": [
            "Social_Isolation", "Peer_Substance_Exposure", "Social_Support",
            "Social_Pressure", "Peer_Influence",
        ],
        "Family": [
            "Childhood_Adversity", "Family_Conflict",
            "Family_Stability", "Family_Support",
        ],
        "Lifestyle": [
            "Sleep_Quality", "Screen_Time", "Sleep_Duration", "Academic_Work_Stress",
        ],
        "Academic / Work": [
            "Workload_Level", "Academic_Performance",
            "Job_Satisfaction", "School_Work_Absenteeism",
        ],
        "Digital": [
            "Night_Time_Device_Usage",
        ],
        "Substance-related": [
            "Previous_Substance_Exposure", "Substance_Use_Intention",
        ],
        "Attitudinal": [
            "Attitude_Toward_Substance_Use", "Awareness_of_Drug_Related_Risks",
        ],
    }
