"""Domain grouping of the 35 audited signals.

Mirrors the taxonomy used by the Streamlit client (``app/utils/utils.py`` →
``get_feature_domains``) so both clients present the same structure.
"""

from __future__ import annotations

DOMAIN_ORDER: tuple[str, ...] = (
    "Psychological",
    "Social / Peer",
    "Family",
    "Lifestyle",
    "Academic / Work",
    "Behavioural",
    "Substance-related",
    "Attitudinal",
    "Digital",
    "Demographic",
)

DOMAINS: dict[str, tuple[str, ...]] = {
    "Psychological": (
        "Hopelessness_Level",
        "Depression_Score",
        "Stress_Level",
        "Loneliness_Level",
        "Impulsivity",
        "Psychological_Distress",
        "Self_Esteem",
        "Anxiety_Level",
        "Emotional_Regulation",
    ),
    "Behavioural": (
        "Behavioural_Problems",
        "Self_Control",
        "Decision_Making_Control",
    ),
    "Social / Peer": (
        "Social_Isolation",
        "Peer_Substance_Exposure",
        "Social_Support",
        "Social_Pressure",
        "Peer_Influence",
    ),
    "Family": (
        "Childhood_Adversity",
        "Family_Conflict",
        "Family_Stability",
        "Family_Support",
    ),
    "Lifestyle": (
        "Sleep_Quality",
        "Screen_Time",
        "Sleep_Duration",
        "Academic_Work_Stress",
    ),
    "Academic / Work": (
        "Workload_Level",
        "Academic_Performance",
        "Job_Satisfaction",
        "School_Work_Absenteeism",
    ),
    "Digital": ("Night_Time_Device_Usage",),
    "Substance-related": (
        "Previous_Substance_Exposure",
        "Substance_Use_Intention",
    ),
    "Attitudinal": (
        "Attitude_Toward_Substance_Use",
        "Awareness_of_Drug_Related_Risks",
    ),
    "Demographic": ("Financial_Stress",),
}

_DOMAIN_BY_FEATURE: dict[str, str] = {
    feature: domain for domain, features in DOMAINS.items() for feature in features
}


def domain_of(feature: str) -> str:
    """Return the domain a signal belongs to, or ``"Unmapped"``."""
    return _DOMAIN_BY_FEATURE.get(feature, "Unmapped")


def domain_counts(features: list[str]) -> list[dict[str, object]]:
    """Count features per domain, ordered by ``DOMAIN_ORDER``."""
    counts = {domain: 0 for domain in DOMAIN_ORDER}
    for feature in features:
        domain = domain_of(feature)
        counts[domain] = counts.get(domain, 0) + 1
    return [
        {"domain": domain, "count": count}
        for domain, count in counts.items()
        if count
    ]


def humanise(feature: str) -> str:
    """``Previous_Substance_Exposure`` → ``Previous Substance Exposure``."""
    return feature.replace("_", " ")
