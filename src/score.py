"""Score current_claims.csv with the trained model and write
predictions_current_claims.csv, sorted by denial_probability descending.

Usage:
    python src/score.py --model_path outputs/model.pkl --data_path data/current_claims.csv --out predictions_current_claims.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_feature_matrix, get_feature_names, load_current
from src.risk_factors import top_risk_factors_for_row
from src.shap_utils import compute_shap_values


def assign_risk_tier(prob: float, high_threshold: float, medium_threshold: float) -> str:
    if prob >= high_threshold:
        return "High"
    if prob >= medium_threshold:
        return "Medium"
    return "Low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/model.pkl")
    parser.add_argument("--data_path", default="data/current_claims.csv")
    parser.add_argument("--out", default="predictions_current_claims.csv")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    high_threshold = artifact["high_threshold"]
    medium_threshold = artifact["medium_threshold"]

    df = load_current(args.data_path)
    X_raw = get_feature_matrix(df)
    X = preprocessor.transform(X_raw)

    denial_probability = model.predict_proba(X)[:, 1]
    predicted_denial = (denial_probability >= high_threshold).astype(int)
    risk_tier = [assign_risk_tier(p, high_threshold, medium_threshold) for p in denial_probability]

    feature_names = get_feature_names(preprocessor)
    shap_values = compute_shap_values(artifact, X)

    top_risk_factors = []
    raw_rows = X_raw.to_dict(orient="records")
    for i in range(len(df)):
        phrases = top_risk_factors_for_row(feature_names, shap_values[i], raw_rows[i], top_n=3)
        top_risk_factors.append("; ".join(phrases) if phrases else "no strong risk drivers identified")

    result = pd.DataFrame(
        {
            "claim_id": df["claim_id"],
            "denial_probability": denial_probability,
            "predicted_denial": predicted_denial,
            "risk_tier": risk_tier,
            "top_risk_factors": top_risk_factors,
        }
    )
    result["explanation"] = ""  # populated for the top 10 by explain.py
    result = result.sort_values("denial_probability", ascending=False).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True) if Path(args.out).parent != Path(".") else None
    result.to_csv(args.out, index=False)

    tier_counts = result["risk_tier"].value_counts()
    print(f"Scored {len(result)} current claims -> {args.out}")
    print(f"Risk tier distribution: {tier_counts.to_dict()}")
    print(f"Top 5 highest-risk claims:\n{result.head(5)[['claim_id', 'denial_probability', 'risk_tier']]}")


if __name__ == "__main__":
    main()
