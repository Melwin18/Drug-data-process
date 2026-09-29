"""Live inference for the web client.

Loads the leakage-free sklearn pipeline produced by this project's training
pipeline and returns the risk estimate, the class probabilities, the per-signal
contributions and non-diagnostic guidance. No retraining happens here.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

from .catalog import feature_specs, metadata, selected_features
from .contributions import linear_contributions, summarise
from .guidance import TARGET_CAVEAT, guidance_for
from .paths import MODEL_PIPELINE_FILE

BAND_BY_RISK: dict[str, int] = {"Low Risk": 1, "Moderate Risk": 2, "High Risk": 3}

_lock = threading.Lock()
_pipeline: Any = None


class InputError(ValueError):
    """Raised when submitted values fail the audited feature specification."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("Invalid input: " + "; ".join(problems))
        self.problems = problems


class ModelUnavailableError(RuntimeError):
    """Raised when the serialised pipeline cannot be loaded."""


def pipeline() -> Any:
    """Load and cache the fitted pipeline (thread-safe, loaded once)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is None:
            if not MODEL_PIPELINE_FILE.exists():
                raise ModelUnavailableError(f"Missing model file: {MODEL_PIPELINE_FILE}")
            try:
                _pipeline = joblib.load(MODEL_PIPELINE_FILE)
            except Exception as error:  # noqa: BLE001 - surfaced to the client
                raise ModelUnavailableError(
                    f"Could not deserialise the pipeline: {error}"
                ) from error
            _apply_sklearn_compat(_pipeline)
    return _pipeline


def _apply_sklearn_compat(model: Any) -> None:
    """Bridge pickles saved by a newer scikit-learn than the one running here.

    The serialised ``LogisticRegression`` was fitted with scikit-learn 1.9, which
    dropped the ``multi_class`` attribute. Older releases (e.g. 1.7) still read it
    inside ``predict_proba``, so an unpickled estimator raises ``AttributeError``
    and the client silently loses its class probabilities. Restoring the attribute
    keeps the probabilities available without retraining or altering the model.
    """
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None and not hasattr(classifier, "multi_class"):
        try:
            classifier.multi_class = "multinomial"
        except (AttributeError, ValueError):
            pass


def default_values() -> dict[str, Any]:
    """Median / first-category profile for every audited signal."""
    specs = feature_specs()
    values: dict[str, Any] = {}
    for feature in selected_features():
        spec = specs.get(feature, {})
        if spec.get("type") == "categorical":
            options = spec.get("values", [])
            values[feature] = options[0] if options else ""
        else:
            values[feature] = spec.get("median", 0.0)
    return values


def validate(values: dict[str, Any]) -> dict[str, Any]:
    """Return cleaned values, or raise :class:`InputError` listing every problem."""
    specs = feature_specs()
    cleaned: dict[str, Any] = {}
    problems: list[str] = []

    for feature in selected_features():
        spec = specs.get(feature, {})
        if feature not in values:
            problems.append(f"{feature}: missing")
            continue

        raw = values[feature]
        if spec.get("type") == "categorical":
            options = spec.get("values", [])
            if raw not in options:
                problems.append(f"{feature}: '{raw}' is not one of {options}")
            else:
                cleaned[feature] = raw
            continue

        try:
            number = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{feature}: '{raw}' is not numeric")
            continue

        low = float(spec.get("min", number))
        high = float(spec.get("max", number))
        if number < low or number > high:
            problems.append(f"{feature}: {number} is outside {low}-{high}")
        else:
            cleaned[feature] = number

    if problems:
        raise InputError(problems)
    return cleaned


def _class_labels(model: Any) -> list[str]:
    """Labels in the order the fitted classifier scores them."""
    classifier = getattr(model, "named_steps", {}).get("classifier", model)
    classes = list(getattr(classifier, "classes_", []))
    target_classes = list(metadata().get("target_classes", []))

    if classes and all(isinstance(label, str) for label in classes):
        return classes

    labels: list[str] = []
    for index in classes:
        try:
            labels.append(target_classes[int(index)])
        except (IndexError, ValueError, TypeError):
            labels.append(str(index))
    return labels or target_classes


def _probabilities(model: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Class probabilities, highest first. Empty when the model cannot score them."""
    if not hasattr(model, "predict_proba"):
        return []
    try:
        raw = model.predict_proba(frame)[0]
    except (AttributeError, RuntimeError, ValueError):
        return []
    rows = [
        {"risk": label, "probability": float(probability)}
        for label, probability in zip(_class_labels(model), raw)
    ]
    rows.sort(key=lambda row: row["probability"], reverse=True)
    return rows


def _drivers(model: Any, frame: pd.DataFrame, predicted_label: str) -> list[dict[str, Any]]:
    """Linear contributions toward the predicted class, best-effort."""
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        labels = _class_labels(model)
        class_index = labels.index(predicted_label) if predicted_label in labels else 0
        rows = linear_contributions(
            preprocessor,
            classifier,
            frame,
            list(selected_features()),
            class_index,
        )
    except (AttributeError, KeyError, ValueError, TypeError):
        return []
    return summarise(rows, limit=8)


def predict(values: dict[str, Any]) -> dict[str, Any]:
    """Estimate the risk category for one audited profile."""
    cleaned = validate(values)
    model = pipeline()
    frame = pd.DataFrame([cleaned])[list(selected_features())]

    raw = model.predict(frame)[0]
    labels = _class_labels(model)
    label = raw if isinstance(raw, str) else (labels[int(raw)] if labels else str(raw))

    probabilities = _probabilities(model, frame)
    confidence = next(
        (row["probability"] for row in probabilities if row["risk"] == label),
        None,
    )

    return {
        "risk": label,
        "band": BAND_BY_RISK.get(label, 0),
        "confidence": confidence,
        "probabilities": probabilities,
        "drivers": _drivers(model, frame, label),
        "guidance": guidance_for(label),
        "model_name": metadata().get("model_name"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": TARGET_CAVEAT,
    }
