"""Per-signal contribution of a single input row.

The final model is a linear classifier behind a preprocessing pipeline, so the
contribution of each transformer output to a class logit is exactly
``coefficient x transformed value``. This module maps those transformer outputs
back to the 35 source signals and aggregates them, giving an honest, per-input
explanation without re-training a surrogate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PREFIXES = ("num__", "cat__")


def transformer_output_names(preprocessor: Any) -> list[str]:
    """Feature names produced by the fitted ``ColumnTransformer``."""
    if hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out()]
    raise AttributeError("Preprocessor does not expose get_feature_names_out()")


def source_feature(transformed_name: str, source_features: list[str]) -> str | None:
    """Map ``cat__Attitude_Toward_Substance_Use_Accepting`` → the source signal.

    One-hot columns are prefixed with their source feature name, so the longest
    source feature that the stripped name starts with is the owner.
    """
    stripped = transformed_name
    for prefix in PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break

    best: str | None = None
    for feature in source_features:
        if stripped == feature or stripped.startswith(feature):
            if best is None or len(feature) > len(best):
                best = feature
    return best


def dense_row(matrix: Any, row: int = 0) -> np.ndarray:
    """Row ``row`` of a possibly-sparse transformed matrix, as a dense array."""
    subset = matrix[row]
    if hasattr(subset, "toarray"):
        subset = subset.toarray()
    return np.asarray(subset, dtype=float).ravel()


def linear_contributions(
    preprocessor: Any,
    classifier: Any,
    input_frame: Any,
    source_features: list[str],
    class_index: int,
) -> list[dict[str, Any]]:
    """Signed contributions per source signal for ``class_index``.

    Positive values push the row toward the chosen class, negative values pull
    it away. Returns rows sorted by absolute contribution, largest first.
    """
    transformed = preprocessor.transform(input_frame)
    values = dense_row(transformed)

    coefficients = np.asarray(classifier.coef_, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    coefficients = coefficients[class_index]

    per_column = coefficients * values
    totals: dict[str, float] = {feature: 0.0 for feature in source_features}

    for name, contribution in zip(transformer_output_names(preprocessor), per_column):
        owner = source_feature(name, source_features)
        if owner is not None:
            totals[owner] += float(contribution)

    rows = [
        {
            "name": feature,
            "contribution": contribution,
            "weight": abs(contribution),
        }
        for feature, contribution in totals.items()
    ]
    rows.sort(key=lambda row: row["weight"], reverse=True)
    return rows


def summarise(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Top ``limit`` contributions with a direction flag and a share of the mass."""
    top = rows[:limit]
    mass = sum(row["weight"] for row in rows) or 1.0
    return [
        {
            "name": row["name"],
            "contribution": round(row["contribution"], 4),
            "direction": "raises" if row["contribution"] > 0 else "lowers",
            "share": round(row["weight"] / mass, 4),
        }
        for row in top
    ]
