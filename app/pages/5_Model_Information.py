import json
import os
import sys

import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils.utils import load_metadata

st.set_page_config(page_title="Model Information", page_icon="[I]", layout="wide")
st.title("Model Information")

metadata = load_metadata()
if "error" in metadata:
    st.error(f"Could not load model metadata: {metadata['error']}")
else:
    st.subheader("Model")
    st.write(f"**Best model:** {metadata.get('model_name', 'N/A')}")
    st.write(f"**Predictors:** {len(metadata.get('selected_features', []))}")
    st.write(f"**Target classes:** {', '.join(metadata.get('target_classes', []))}")
    st.write(f"**Training date:** {metadata.get('training_date', 'N/A')}")

st.subheader("Methodology")
st.markdown(
    "- Stratified 80/20 train-test split with random seed 42.\n"
    "- Numerical missing values imputed with the training median and scaled.\n"
    "- Categorical values imputed with the training mode and one-hot encoded.\n"
    "- Logistic Regression selected using training data and cross-validation.\n"
    "- The held-out test set was reserved for final evaluation."
)

st.subheader("Limitations")
st.markdown(
    "- The target is an operational research label, not a clinically validated diagnosis.\n"
    "- The data is cross-sectional and self-reported, so response bias is possible.\n"
    "- Observational feature importance does not establish causality.\n"
    "- The model cannot diagnose addiction or guarantee future drug use."
)

st.warning("This application is for academic and research demonstration only. Consult qualified professionals for health-related decisions.")
