import os
import sys

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils.utils import load_metadata

st.set_page_config(page_title="Model Performance", page_icon="[P]", layout="wide")
st.title("Model Performance")
st.write("Evaluation results from the held-out test set and saved training reports.")

metadata = load_metadata()
metrics = metadata.get("evaluation_metrics", {}) if "error" not in metadata else {}
if not metrics:
    st.error("Model metrics could not be loaded.")
else:
    columns = st.columns(5)
    for column, label in zip(columns, ["Accuracy", "Macro F1", "Macro Precision", "Balanced Accuracy", "ROC-AUC"]):
        value = metrics.get(label, 0)
        column.metric(label, f"{value * 100:.1f}%" if label == "Accuracy" else f"{value:.3f}")

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
comparison_path = os.path.join(root, "results", "final_model_comparison.csv")
if os.path.exists(comparison_path):
    st.subheader("Model Comparison")
    st.dataframe(pd.read_csv(comparison_path), hide_index=True, use_container_width=True)

st.subheader("Comparison Charts")
chart_names = [
    ("Macro F1 Comparison", "macro_f1_comparison.png"),
    ("Accuracy Comparison", "model_accuracy_comparison.png"),
    ("ROC-AUC Comparison", "roc_auc_comparison.png"),
    ("Balanced Accuracy Comparison", "balanced_accuracy_comparison.png"),
]
for title, filename in chart_names:
    path = os.path.join(root, "results", filename)
    if os.path.exists(path):
        st.image(path, caption=title, use_container_width=True)

st.subheader("Confusion Matrices")
confusion_dir = os.path.join(root, "results", "confusion_matrices")
images = [name for name in sorted(os.listdir(confusion_dir)) if name.endswith(".png")] if os.path.isdir(confusion_dir) else []
if images:
    columns = st.columns(2)
    for index, image in enumerate(images):
        columns[index % 2].image(os.path.join(confusion_dir, image), caption=image, use_container_width=True)
else:
    st.info("No confusion matrix images were found.")

st.subheader("Per-Class Classification Reports")
reports_dir = os.path.join(root, "results", "classification_reports")
report_files = [name for name in sorted(os.listdir(reports_dir)) if name.endswith(".txt")] if os.path.isdir(reports_dir) else []
for report_file in report_files:
    with st.expander(report_file.replace("_report.txt", "")):
        with open(os.path.join(reports_dir, report_file), encoding="utf-8") as file:
            st.text(file.read())
