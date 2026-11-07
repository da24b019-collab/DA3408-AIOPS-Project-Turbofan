"""
From validated readings to one prediction per engine.

    readings -> 42 features per row (same code as training, so no skew)
             -> keep each engine's LAST cycle (the "now" we predict for)
             -> probability -> at_risk / risk level -> explanation
"""

import pandas as pd

from src.features.feature_logic import compute_features_pandas

TOP_FACTORS = 3


def risk_level(probability: float, threshold: float) -> str:
    """high = at risk; medium = at least half the threshold; low = below that."""
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def summary_sentence(level: str, factors: list) -> str:
    """One plain-language sentence for the operator."""
    if level == "low":
        return "Low risk: the recent sensor readings look like a healthy engine."
    opening = ("High risk of failure within 30 cycles." if level == "high"
               else "Medium risk: keep this engine under watch.")
    raising = [f["description"] for f in factors if f["effect"] == "raises risk"]
    if not raising:
        return opening
    if len(raising) == 1:
        reasons = f"{raising[0]} is"
    else:  # "A, B and C are"
        reasons = f"{', '.join(raising[:-1])} and {raising[-1]} are"
    return f"{opening} Main reasons: {reasons} raising the risk."


def predict_engines(readings: pd.DataFrame, service, sensors: list, window: int) -> dict:
    """Run the whole prediction for every engine in the readings."""
    features = service.settings.features
    threshold = service.info["threshold"]

    with_features = compute_features_pandas(readings, sensors, window)
    last_rows = with_features.groupby("unit").tail(1).reset_index(drop=True)
    cycles_received = readings.groupby("unit")["cycle"].count()

    probabilities = service.predict_proba(last_rows[features])

    engines = []
    for i, row in last_rows.iterrows():
        unit, prob = int(row["unit"]), float(probabilities[i])
        level = risk_level(prob, threshold)
        factors = service.explain(last_rows.loc[[i], features])[:TOP_FACTORS]
        warnings = []
        if cycles_received[unit] < window:
            warnings.append(
                f"Only {cycles_received[unit]} cycle(s) received; at least {window} give a "
                "more reliable prediction.")
        engines.append({
            "unit": unit,
            "cycles_received": int(cycles_received[unit]),
            "last_cycle": int(row["cycle"]),
            "failure_probability": round(prob, 4),
            "at_risk": prob >= threshold,
            "risk_level": level,
            "explanation": {"summary": summary_sentence(level, factors), "top_factors": factors},
            "warnings": warnings,
        })

    return {
        "model_version": service.info["version"],
        "threshold": round(threshold, 4),
        "n_engines": len(engines),
        "engines": engines,
        "warnings": [],
    }
