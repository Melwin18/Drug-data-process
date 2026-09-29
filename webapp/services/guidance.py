"""Non-diagnostic guidance shown next to an estimate.

Same substance as the recommendations in ``app/utils/utils.py`` (the Streamlit
client), re-voiced for the web client. These are general preventive pointers, not
medical advice, and are never phrased as a diagnosis.
"""

from __future__ import annotations

GUIDANCE: dict[str, list[str]] = {
    "Low Risk": [
        "Keep the routine that is working — sleep, movement, people.",
        "Stay connected to the support you already have.",
        "Watch the load. Most trajectories bend before they break.",
    ],
    "Moderate Risk": [
        "Talk to a counsellor about the stressors behind the score.",
        "Rebuild the support links that have gone quiet.",
        "Handle the emotional load now instead of later.",
        "Review sleep, screen time and physical activity.",
    ],
    "High Risk": [
        "Speak to a qualified healthcare or mental-health professional promptly.",
        "Counselling and evidence-based support programmes are available and effective.",
        "In immediate danger, contact emergency or crisis services.",
        "Peer support and community resources can carry part of the weight.",
    ],
}

TARGET_CAVEAT = (
    "Drug_Abuse_Risk is an operational research label, not a clinically validated "
    "diagnosis. The estimate does not prove future drug use."
)


def guidance_for(risk_label: str) -> list[str]:
    """Return the guidance rows for a predicted label (empty list if unknown)."""
    return GUIDANCE.get(risk_label, [])
