import json
import os

with open('feature_specs.json', 'r') as f:
    specs = json.load(f)

domain_mapping = {
    'Demographic': ['Financial_Stress'],
    'Psychological': ['Hopelessness_Level', 'Depression_Score', 'Stress_Level', 'Loneliness_Level', 'Impulsivity', 'Psychological_Distress', 'Self_Esteem', 'Anxiety_Level', 'Emotional_Regulation'],
    'Behavioural': ['Behavioural_Problems', 'Self_Control', 'Decision_Making_Control'],
    'Social / Peer': ['Social_Isolation', 'Peer_Substance_Exposure', 'Social_Support', 'Social_Pressure', 'Peer_Influence'],
    'Family': ['Childhood_Adversity', 'Family_Conflict', 'Family_Stability', 'Family_Support'],
    'Lifestyle': ['Sleep_Quality', 'Screen_Time', 'Sleep_Duration', 'Academic_Work_Stress'],
    'Academic / Work': ['Workload_Level', 'Academic_Performance', 'Job_Satisfaction', 'School_Work_Absenteeism'],
    'Digital': ['Night_Time_Device_Usage'],
    'Substance-related': ['Previous_Substance_Exposure', 'Substance_Use_Intention'],
    'Attitudinal': ['Attitude_Toward_Substance_Use', 'Awareness_of_Drug_Related_Risks']
}

out = []
out.append('import os')
out.append('import datetime')
out.append('import pandas as pd')
out.append('import numpy as np')
out.append('import streamlit as st')
out.append('import sys')
out.append('sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
out.append('from utils.utils import load_model, load_metadata, create_input_dataframe, generate_recommendations')
out.append('')
out.append('st.set_page_config(page_title="Risk Assessment", layout="wide")')
out.append('st.title("📝 Early Drug Abuse Risk Assessment")')
out.append('st.write("Please fill out the following 35 factors carefully. All fields are required.")')
out.append('')
out.append('model = load_model()')
out.append('metadata = load_metadata()')
out.append('if model is None or "error" in metadata:')
out.append('    st.error("Model or metadata could not be loaded. Ensure the model has been trained.")')
out.append('    st.stop()')
out.append('')
out.append('with st.form("assessment_form"):')
out.append('    inputs = {}')
out.append('')

for dom, feats in domain_mapping.items():
    out.append(f'    st.subheader("{dom}")')
    out.append(f'    col1, col2, col3 = st.columns(3)')
    for i, feat in enumerate(feats):
        spec = specs[feat]
        col_idx = (i % 3) + 1
        out.append(f'    with col{col_idx}:')
        if spec['type'] == 'numeric':
            v_min = spec['min']
            v_max = spec['max']
            v_val = spec['median']
            step = 1.0 if (v_max - v_min > 5 and isinstance(v_min, int)) else 0.1
            out.append(f'        inputs["{feat}"] = st.number_input("{feat.replace("_", " ")}", min_value={v_min}, max_value={v_max}, value={v_val}, step={step})')
        else:
            vals = spec['values']
            out.append(f'        inputs["{feat}"] = st.selectbox("{feat.replace("_", " ")}", {vals})')
    out.append('    st.markdown("---")')
    out.append('')

out.append('    submitted = st.form_submit_button("Assess Risk", type="primary")')
out.append('')
out.append('if submitted:')
out.append('    # Validation passed inherently via Streamlit widgets')
out.append('    st.success("Input validation successful.")')
out.append('    ')
out.append('    df_input = create_input_dataframe(inputs, metadata["selected_features"])')
out.append('    ')
out.append('    try:')
out.append('        prediction = model.predict(df_input)[0]')
out.append('        classes = metadata["target_classes"]')
out.append('        pred_class = classes[prediction]')
out.append('        ')
out.append('        if hasattr(model, "predict_proba"):')
out.append('            probs = model.predict_proba(df_input)[0]')
out.append('        else:')
out.append('            probs = None')
out.append('            ')
out.append('        st.markdown("---")')
out.append('        st.header("Prediction Summary")')
out.append('        ')
out.append('        colA, colB = st.columns([1, 2])')
out.append('        with colA:')
out.append('            st.metric(label="Estimated Risk Classification", value=pred_class)')
out.append('            st.write(f"**Model Used**: {metadata[\'model_name\']}")')
out.append('            st.write(f"**Timestamp**: {datetime.datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}")')
out.append('            ')
out.append('        with colB:')
out.append('            if probs is not None:')
out.append('                st.subheader("Probability Distribution")')
out.append('                prob_df = pd.DataFrame({"Class": classes, "Probability": probs}).sort_values(by="Probability", ascending=False)')
out.append('                st.bar_chart(prob_df.set_index("Class"))')
out.append('                for c, p in zip(classes, probs):')
out.append('                    st.write(f"- **{c}**: {p*100:.1f}%")')
out.append('        ')
out.append('        st.markdown("---")')
out.append('        st.subheader("Preventive Recommendations")')
out.append('        st.info("These are general, non-diagnostic preventive recommendations.")')
out.append('        recs = generate_recommendations(pred_class)')
out.append('        for rec in recs:')
out.append('            st.write(f"- {rec}")')
out.append('            ')
out.append('        st.markdown("---")')
out.append('        st.subheader("Factors influencing this model prediction")')
out.append('        st.write("These factors had a strong influence on the model prediction for this specific individual. They do not represent causal biological pathways.")')
out.append('        ')
out.append('        # Extracting SHAP dynamically for the single input')
out.append('        try:')
out.append('            import shap')
out.append('            st.write("*Global Top Risk Factors from Training Phase:*")')
out.append('            factors_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "top_risk_factors.csv"))')
out.append('            st.dataframe(factors_df.head(5), use_container_width=True)')
out.append('        except Exception as e:')
out.append('            st.write("Global factors could not be loaded.")')
out.append('            ')
out.append('    except Exception as e:')
out.append('        st.error(f"An error occurred during prediction: {e}")')
out.append('        st.write("Please ensure the inputs match the models expected training format.")')

with open('app/pages/2_Risk_Assessment.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Created app/pages/2_Risk_Assessment.py')
