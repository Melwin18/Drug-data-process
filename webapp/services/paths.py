"""Filesystem anchors for the web client.

Everything resolves from this file's location so the server works no matter
which directory it is launched from.
"""

from __future__ import annotations

from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = SERVICES_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent

PUBLIC_DIR = WEBAPP_DIR / "public"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PIPELINE_FILE = MODELS_DIR / "best_model_pipeline.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
FEATURE_SPECS_FILE = PROJECT_ROOT / "feature_specs.json"

TOP_RISK_FACTORS_FILE = RESULTS_DIR / "top_risk_factors.csv"
COMBINED_IMPORTANCE_FILE = RESULTS_DIR / "combined_importance_v2.csv"
MODEL_COMPARISON_FILE = RESULTS_DIR / "final_model_comparison.csv"
FEATURE_SELECTION_FILE = RESULTS_DIR / "feature_selection.csv"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
