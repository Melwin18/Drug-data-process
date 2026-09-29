import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils.utils import (
    create_input_dataframe,
    generate_recommendations,
    get_prediction_and_proba,
    load_feature_specs,
    load_metadata,
    load_model,
)

st.set_page_config(page_title="Risk Assessment", page_icon="[R]", layout="wide")
st.title("Risk Assessment")
st.write("Enter the 35 feature values, then submit the form for an ML-estimated risk category.")
st.caption("No personal identifying information is collected or stored by this application.")

model, model_error = load_model()
metadata = load_metadata()
specs = load_feature_specs()

if model_error:
    st.error(f"Model could not be loaded: {model_error}")
    st.stop()
if "error" in metadata:
    st.error(f"Model metadata could not be loaded: {metadata['error']}")
    st.stop()
if "error" in specs:
    st.error(f"Feature specifications could not be loaded: {specs['error']}")
    st.stop()

features = metadata.get("selected_features", [])
missing = [feature for feature in features if feature not in specs]
if missing:
    st.error(f"Missing feature specifications: {', '.join(missing)}")
    st.stop()

feature_domains = {
    "Demographic": ["Financial_Stress"],
    "Psychological": ["Hopelessness_Level", "Depression_Score", "Stress_Level", "Loneliness_Level", "Impulsivity", "Psychological_Distress", "Self_Esteem", "Anxiety_Level", "Emotional_Regulation"],
    "Behavioural": ["Behavioural_Problems", "Self_Control", "Decision_Making_Control"],
    "Social / Peer": ["Social_Isolation", "Peer_Substance_Exposure", "Social_Support", "Social_Pressure", "Peer_Influence"],
    "Family": ["Childhood_Adversity", "Family_Conflict", "Family_Stability", "Family_Support"],
    "Lifestyle": ["Sleep_Quality", "Screen_Time", "Sleep_Duration", "Academic_Work_Stress"],
    "Academic / Work": ["Workload_Level", "Academic_Performance", "Job_Satisfaction", "School_Work_Absenteeism"],
    "Digital": ["Night_Time_Device_Usage"],
    "Substance-related": ["Previous_Substance_Exposure", "Substance_Use_Intention"],
    "Attitudinal": ["Attitude_Toward_Substance_Use", "Awareness_of_Drug_Related_Risks"],
}

with st.form("risk_assessment_form"):
    inputs = {}
    for domain, domain_features in feature_domains.items():
        st.subheader(domain)
        columns = st.columns(3)
        for index, feature in enumerate(domain_features):
            if feature not in specs:
                continue
            spec = specs[feature]
            label = feature.replace("_", " ")
            with columns[index % 3]:
                if spec["type"] == "numeric":
                    inputs[feature] = st.number_input(
                        label,
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(spec["median"]),
                        step=0.1,
                    )
                else:
                    inputs[feature] = st.selectbox(label, spec["values"])
    submitted = st.form_submit_button("Assess Risk", type="primary")

if submitted:
    try:
        input_df = create_input_dataframe(inputs, features)
        prediction, probabilities, probability_classes = get_prediction_and_proba(
            model, input_df, metadata["target_classes"]
        )
        st.success("Assessment completed.")
        st.header("Assessment Result")
        st.metric("ML-Estimated Risk Category", prediction)
        st.caption(f"Model: {metadata.get('model_name', 'N/A')} | Generated: {datetime.now():%Y-%m-%d %H:%M}")

        if probabilities is not None:
            classes = probability_classes or metadata["target_classes"]
            probability_df = pd.DataFrame({"Risk Category": classes, "Probability": probabilities})
            probability_df["Probability (%)"] = probability_df["Probability"] * 100
            st.subheader("Probability Distribution")
            st.bar_chart(probability_df.set_index("Risk Category")["Probability (%)"])
            st.dataframe(probability_df[["Risk Category", "Probability (%)"]], hide_index=True, use_container_width=True)

        st.subheader("General Preventive Recommendations")
        st.caption("These are non-diagnostic guidance points, not medical advice.")
        for recommendation in generate_recommendations(prediction):
            st.write(f"- {recommendation}")

        factors_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "top_risk_factors.csv")
        if os.path.exists(factors_path):
            st.subheader("Global Model-Influential Features")
            st.caption("These are global training-phase importance scores and do not establish causation.")
            st.dataframe(pd.read_csv(factors_path).head(10), hide_index=True, use_container_width=True)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
