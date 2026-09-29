import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feature Importance", page_icon="[F]", layout="wide")
st.title("Feature Importance and Interpretability")
st.write("Global feature importance summarizes which inputs influenced model predictions during training.")
st.warning("Importance scores describe model behaviour and do not prove biological or psychosocial causation.")

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
importance_path = os.path.join(root, "results", "top_risk_factors.csv")
if os.path.exists(importance_path):
    st.subheader("Top Risk Factors")
    st.dataframe(pd.read_csv(importance_path), hide_index=True, use_container_width=True)

for title, filename in [
    ("Combined Feature Importance", "feature_importance_combined.png"),
    ("Random Forest Importance", "rf_importance_v2.png"),
    ("XGBoost Importance", "xgb_importance_v2.png"),
    ("Permutation Importance", "permutation_importance_v2.png"),
    ("Feature Correlation Heatmap", "correlation_heatmap_v2.png"),
]:
    path = os.path.join(root, "results", filename)
    if os.path.exists(path):
        st.image(path, caption=title, use_container_width=True)
