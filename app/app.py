import streamlit as st

st.set_page_config(
    page_title="Early Drug Abuse Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar info
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
st.sidebar.info("Use the pages above to navigate the application.")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**AI-Based Early Drug Abuse  
Risk Assessment System**

*Academic Research Prototype*  
ML Model: Logistic Regression  
Features: 35 | Dataset: 5,000 rows
""")

# ---------------------------------------------------------------------------
# Main welcome content
# ---------------------------------------------------------------------------
st.title("🛡️ Early Drug Abuse Risk Assessment")
st.markdown("### AI-Based Early Warning System Using Machine Learning")

st.markdown("""
Welcome to the academic prototype application for estimating drug-abuse risk using a 
machine-learning early warning system.

Use the **sidebar** to navigate between pages:

| Page | Description |
|---|---|
| **Home** | Project overview and important disclaimers |
| **Risk Assessment** | Interactive 35-feature prediction form |
| **Model Performance** | Metrics and visualisations from the training phase |
| **Feature Importance** | SHAP-based explanations of influential factors |
| **Model Information** | Methodology, limitations, and technical details |
""")

st.warning("""
⚠️ **IMPORTANT DISCLAIMER**

This application is for **academic and research demonstration only**.  
It provides a statistical risk estimate — it **is NOT a medical diagnostic system**  
and cannot determine whether a person is currently using drugs or will develop addiction.  
Please consult a qualified healthcare or mental health professional for advice or diagnosis.
""")
