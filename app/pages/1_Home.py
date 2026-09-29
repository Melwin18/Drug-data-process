import os
import sys
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils.utils import load_metadata

st.set_page_config(page_title="Home - Drug Abuse Risk Assessment", page_icon="[H]", layout="wide")

st.title("AI-Based Early Drug Abuse Risk Assessment")
st.subheader("Early Warning System Using Machine Learning")
st.markdown("---")

st.header("Project Objective")
st.write(
    "This academic prototype estimates an ML-based risk category from 35 audited "
    "behavioural, psychological, social, family, lifestyle, academic, digital, "
    "and attitudinal factors."
)

metadata = load_metadata()
if "error" in metadata:
    st.error(f"Model metadata could not be loaded: {metadata['error']}")
else:
    metrics = metadata.get("evaluation_metrics", {})
    st.header("Model at a Glance")
    columns = st.columns(4)
    columns[0].metric("Best Model", metadata.get("model_name", "N/A"))
    columns[1].metric("Accuracy", f"{metrics.get('Accuracy', 0) * 100:.1f}%")
    columns[2].metric("Macro F1", f"{metrics.get('Macro F1', 0):.3f}")
    columns[3].metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.3f}")

st.warning(
    "This application is for academic and research demonstration only. "
    "It is not a medical diagnostic system and cannot determine current or future drug use. "
    "Consult a qualified healthcare or mental-health professional for advice."
)

st.info("Use the sidebar to open the risk assessment, performance, feature importance, and model information pages.")
