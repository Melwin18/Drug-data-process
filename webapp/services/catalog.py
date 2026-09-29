"""Read-only views over the trained model's artefacts.

Every number the web client displays comes from a file the training pipeline
produced — the servable metadata, ``feature_specs.json``, the audit reports in
``reports/`` and the result tables in ``results/``. Nothing here is invented.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from typing import Any

from .domains import domain_counts, domain_of, humanise
from .paths import (
    CLASSIFICATION_REPORTS_DIR,
    COMBINED_IMPORTANCE_FILE,
    FEATURE_SELECTION_FILE,
    FEATURE_SPECS_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_METADATA_FILE,
    PROJECT_ROOT,
    TOP_RISK_FACTORS_FILE,
)

CORPUS_REPORT_FILE = PROJECT_ROOT / "reports" / "final_feature_set.txt"

METRIC_KEYS: tuple[str, ...] = (
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Balanced Accuracy",
    "ROC-AUC",
)


class ArtefactError(RuntimeError):
    """Raised when a required artefact is missing or unreadable."""


@lru_cache(maxsize=1)
def metadata() -> dict[str, Any]:
    """Servable model metadata (``models/model_metadata.json``)."""
    if not MODEL_METADATA_FILE.exists():
        raise ArtefactError(f"Missing model metadata: {MODEL_METADATA_FILE}")
    with MODEL_METADATA_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def feature_specs() -> dict[str, dict[str, Any]]:
    """Per-feature input specification (``feature_specs.json``)."""
    if not FEATURE_SPECS_FILE.exists():
        raise ArtefactError(f"Missing feature specs: {FEATURE_SPECS_FILE}")
    with FEATURE_SPECS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def selected_features() -> tuple[str, ...]:
    """The 35 predictors, in training column order."""
    return tuple(metadata().get("selected_features", ()))


@lru_cache(maxsize=1)
def _shap_scores() -> dict[str, float]:
    return _read_float_column(TOP_RISK_FACTORS_FILE, "Original_Feature", "SHAP_Importance")


@lru_cache(maxsize=1)
def _combined_scores() -> dict[str, float]:
    return _read_float_column(COMBINED_IMPORTANCE_FILE, "Feature", "Combined_Score")


@lru_cache(maxsize=1)
def _selection_scores() -> dict[str, float]:
    return _read_float_column(FEATURE_SELECTION_FILE, "Feature", "Importance")


def _read_float_column(path, key_column: str, value_column: str) -> dict[str, float]:
    """Read ``path`` as CSV into ``{key: float(value)}``, tolerant of format drift."""
    if not path.exists():
        return {}
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                scores[row[key_column]] = float(row[value_column])
            except (KeyError, TypeError, ValueError):
                continue
    return scores


def feature_catalog() -> list[dict[str, Any]]:
    """The 35 signals with spec, domain and training-phase importance."""
    specs = feature_specs()
    shap = _shap_scores()
    combined = _combined_scores()

    rows: list[dict[str, Any]] = []
    for feature in selected_features():
        spec = specs.get(feature, {})
        row: dict[str, Any] = {
            "name": feature,
            "label": humanise(feature),
            "domain": domain_of(feature),
            "type": spec.get("type", "numeric"),
            "min": spec.get("min"),
            "max": spec.get("max"),
            "median": spec.get("median"),
            "values": spec.get("values", []),
            "shap": shap.get(feature),
            "combined": combined.get(feature),
        }
        rows.append(row)
    return rows


def importance_rows() -> list[dict[str, Any]]:
    """Signals ranked by held-out SHAP importance, highest first."""
    rows = [
        {
            "name": feature,
            "label": humanise(feature),
            "domain": domain_of(feature),
            "shap": _shap_scores().get(feature, 0.0),
            "combined": _combined_scores().get(feature),
            "selection": _selection_scores().get(feature),
        }
        for feature in selected_features()
    ]
    rows.sort(key=lambda row: row["shap"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def domain_rows() -> list[dict[str, Any]]:
    """Signal counts per domain for the 35-feature set."""
    return domain_counts(list(selected_features()))


@lru_cache(maxsize=1)
def performance() -> dict[str, Any]:
    """Held-out metrics, model-comparison rows and per-class reports."""
    meta = metadata()
    metrics = meta.get("evaluation_metrics", {})
    return {
        "model_name": meta.get("model_name"),
        "metrics": [
            {
                "key": key,
                "label": key,
                "value": metrics.get(key),
                "display": _format_metric(key, metrics.get(key)),
            }
            for key in METRIC_KEYS
        ],
        "training_time_seconds": metrics.get("Training Time"),
        "prediction_time_seconds": metrics.get("Prediction Time"),
        "comparison": _model_comparison(),
        "reports": _classification_reports(),
    }


def _format_metric(key: str, value: Any) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if key == "Accuracy" else f"{value:.3f}"


def _model_comparison() -> list[dict[str, Any]]:
    if not MODEL_COMPARISON_FILE.exists():
        return []
    with MODEL_COMPARISON_FILE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "model": row["Model"],
            "accuracy": float(row["Accuracy"]),
            "macro_f1": float(row["Macro F1"]),
            "roc_auc": float(row["ROC-AUC"]),
            "training_time": float(row["Training Time"]),
        }
        for row in rows
    ]


def _classification_reports() -> list[dict[str, str]]:
    if not CLASSIFICATION_REPORTS_DIR.is_dir():
        return []
    reports: list[dict[str, str]] = []
    for path in sorted(CLASSIFICATION_REPORTS_DIR.glob("*_report.txt")):
        reports.append(
            {
                "model": path.stem.replace("_report", "").replace("_", " "),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return reports


@lru_cache(maxsize=1)
def methodology() -> dict[str, Any]:
    """Pipeline facts taken from the training reports, not hand-typed."""
    meta = metadata()
    return {
        "model_name": meta.get("model_name"),
        "baseline_model": meta.get("baseline_model"),
        "ensemble_models": meta.get("ensemble_models", []),
        "ensemble_method": meta.get("ensemble_method"),
        "hyperparameters": meta.get("best_hyperparameters"),
        "training_date": meta.get("training_date"),
        "random_seed": meta.get("random_seed"),
        "target_classes": meta.get("target_classes", []),
        "target_limitation_statement": meta.get("target_limitation_statement"),
        "corpus": _corpus_sizes(),
    }


def _corpus_sizes() -> dict[str, Any]:
    """Split sizes and candidate pool, parsed from ``reports/final_feature_set.txt``."""
    if not CORPUS_REPORT_FILE.exists():
        return {}
    text = CORPUS_REPORT_FILE.read_text(encoding="utf-8")
    patterns = {
        "original_predictors": r"Original feature count\s*:\s*(\d+)",
        "candidate_pool": r"Candidate pool\s*:\s*(\d+)",
        "final_predictors": r"Final predictor count\s*:\s*(\d+)",
        "train_rows": r"Train set size\s*:\s*(\d+)",
        "test_rows": r"Test set size\s*:\s*(\d+)",
    }
    sizes: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            sizes[key] = int(match.group(1))
    sizes["leakage_status"] = "CLEAN" if "LEAKAGE STATUS: CLEAN" in text else "UNKNOWN"
    return sizes


def overview() -> dict[str, Any]:
    """One payload for the dashboard's metric tiles."""
    meta = metadata()
    metrics = meta.get("evaluation_metrics", {})
    corpus = _corpus_sizes()
    return {
        "model_name": meta.get("model_name"),
        "target_classes": meta.get("target_classes", []),
        "training_date": meta.get("training_date"),
        "random_seed": meta.get("random_seed"),
        "hyperparameters": meta.get("best_hyperparameters"),
        "roc_auc": metrics.get("ROC-AUC"),
        "accuracy": metrics.get("Accuracy"),
        "macro_f1": metrics.get("Macro F1"),
        "balanced_accuracy": metrics.get("Balanced Accuracy"),
        "prediction_time_seconds": metrics.get("Prediction Time"),
        "training_time_seconds": metrics.get("Training Time"),
        "feature_count": len(selected_features()),
        "domain_rows": domain_rows(),
        "corpus": corpus,
        "top_signals": importance_rows()[:6],
        "target_limitation_statement": meta.get("target_limitation_statement"),
    }
