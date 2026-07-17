"""Generate plain-English, analyst-facing explanations for the highest-risk
current claims using an LLM, grounded strictly in the claim's field values
and the model's SHAP-derived risk factors.

Usage:
    python src/explain.py --predictions predictions_current_claims.csv --data_path data/current_claims.csv --top_n 10
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = """You are writing short risk notes for a hospital billing analyst who \
reviews flagged insurance claims before they are submitted. The analyst has seconds, \
not minutes, to read each note and decide what to do next.

Rules you must follow:
- Use ONLY the claim facts and risk factors given to you below. Never invent, assume, \
or infer any fact not explicitly provided (no guessing at a diagnosis, a patient detail, \
a policy, or a reason not listed).
- Plain English only. No insurance or billing jargon the analyst would need to look up \
(e.g. avoid terms like "adjudication", "COB", "timely filing" without explanation).
- Include exactly one specific, concrete recommended action the analyst can take right now.
- Explicitly state that this is a risk estimate, not a guarantee the claim will be denied.
- Write exactly 2-3 sentences. No preamble, no headers, no bullet points."""

USER_PROMPT_TEMPLATE = """Claim ID: {claim_id}
Payer type: {payer_type}
Visit type: {visit_type}
Total billed: ${total_billed:,.0f}
Model's predicted denial risk: {denial_probability:.0%} ({risk_tier} risk tier)
Top risk factors identified by the model: {top_risk_factors}

Write the 2-3 sentence risk note for this claim."""


def build_prompt(claim_row: dict) -> str:
    return USER_PROMPT_TEMPLATE.format(
        claim_id=claim_row["claim_id"],
        payer_type=claim_row["payer_type"],
        visit_type=claim_row["visit_type"],
        total_billed=claim_row["total_billed"],
        denial_probability=claim_row["denial_probability"],
        risk_tier=claim_row["risk_tier"],
        top_risk_factors=claim_row["top_risk_factors"],
    )


# Maps a risk-factor phrase (as produced by src/risk_factors.py) to a single
# concrete action. Used only as a fallback when no LLM API access is available,
# so the manually drafted examples still vary claim-to-claim instead of repeating
# one generic template.
ACTION_BY_RISK_FACTOR = {
    "prior authorization is required but not on file": "confirm whether prior authorization was obtained and attach it before submission",
    "a referral is required but not on file": "verify a referral exists and attach it before submission",
    "required supporting documentation appears to be missing": "pull the missing supporting documentation and attach it before submission",
    "patient eligibility was not verified before submission": "re-verify the patient's eligibility with the payer before submission",
    "the provider is out of network for this payer": "confirm the provider's network status with the payer before submission",
}


def draft_explanation(claim_row: dict) -> str:
    """Hand-drafted fallback explanation, used only when the LLM API is unavailable.
    Still grounded in this specific claim's risk factors (not a generic template)."""
    factors = [f.strip() for f in str(claim_row["top_risk_factors"]).split(";")]
    primary = factors[0] if factors else "no strong risk drivers identified"

    if claim_row["risk_tier"] == "Low" or primary not in ACTION_BY_RISK_FACTOR:
        return (
            f"This claim is flagged as {claim_row['risk_tier'].lower()} risk "
            f"({claim_row['denial_probability']:.0%} predicted denial probability); "
            "the model did not find a specific documentation, authorization, or eligibility gap driving this score. "
            "Recommended action: no special pre-submission review needed -- proceed with normal submission. "
            "This is a model-based risk estimate, not a guarantee the claim will be paid."
        )

    action = ACTION_BY_RISK_FACTOR[primary]
    other_factors = factors[1:2]  # mention at most one more, to stay within 2-3 sentences
    factor_clause = primary
    if other_factors:
        factor_clause += f" and {other_factors[0]}"

    return (
        f"This claim is flagged as {claim_row['risk_tier'].lower()} risk "
        f"({claim_row['denial_probability']:.0%} predicted denial probability), "
        f"mainly because {factor_clause}. Recommended action: {action}. "
        "This is a model-based risk estimate, not a guarantee the claim will be denied."
    )


def call_llm(user_prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="predictions_current_claims.csv")
    parser.add_argument("--data_path", default="data/current_claims.csv")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--out_json", default="outputs/explanations_top10.json")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    claims = pd.read_csv(args.data_path)
    merged = predictions.merge(claims[["claim_id", "payer_type", "visit_type", "total_billed"]], on="claim_id")
    merged = merged.sort_values("denial_probability", ascending=False).reset_index(drop=True)

    top_claims = merged.head(args.top_n)
    low_risk_claim = merged.tail(1)  # deliberate sanity check on a low-risk claim

    records = []
    have_key = bool(os.environ.get("OPENAI_API_KEY"))
    llm_failed = False

    for _, row in pd.concat([top_claims, low_risk_claim]).iterrows():
        row_dict = row.to_dict()
        prompt = build_prompt(row_dict)
        source = "llm"
        if have_key and not llm_failed:
            try:
                explanation = call_llm(prompt, model=args.model)
            except Exception as exc:
                print(f"LLM call failed ({exc}); falling back to manually drafted explanations for remaining claims.")
                llm_failed = True
                have_key = False  # stop retrying per-row after the first failure
        if not have_key or llm_failed:
            explanation = draft_explanation(row_dict)
            source = "manual_draft (LLM API unavailable)"
        records.append(
            {
                "claim_id": row_dict["claim_id"],
                "denial_probability": row_dict["denial_probability"],
                "risk_tier": row_dict["risk_tier"],
                "top_risk_factors": row_dict["top_risk_factors"],
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": prompt,
                "explanation": explanation,
                "source": source,
            }
        )
        print(f"{row_dict['claim_id']} ({row_dict['risk_tier']}, {row_dict['denial_probability']:.0%}): {explanation}\n")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved full prompt/response records to {args.out_json}")

    # Merge top-N explanations back into the predictions CSV (low-risk sanity
    # example is intentionally NOT merged -- it's outside the top 10 scope).
    explanation_by_id = {r["claim_id"]: r["explanation"] for r in records[: args.top_n]}
    predictions["explanation"] = predictions["claim_id"].map(explanation_by_id).fillna(predictions["explanation"])
    predictions.to_csv(args.predictions, index=False)
    print(f"Updated {args.predictions} with explanations for top {args.top_n} claims")


if __name__ == "__main__":
    main()
