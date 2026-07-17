"""Turn SHAP values + raw claim fields into human-readable risk-factor phrases.

Used by score.py to populate `top_risk_factors` and to ground the LLM prompt
in explain.py -- the LLM is only ever shown these grounded phrases, not raw
column names or SHAP internals.
"""

import numpy as np

# One human phrase per base feature. `{value}` is filled in for numeric features.
BINARY_PHRASES = {
    "auth_gap": {1: "prior authorization is required but not on file"},
    "referral_gap": {1: "a referral is required but not on file"},
    "missing_documentation_flag": {1: "required supporting documentation appears to be missing"},
    "eligibility_verified": {0: "patient eligibility was not verified before submission"},
    "is_in_network": {0: "the provider is out of network for this payer"},
    "prior_auth_required": {},  # covered by auth_gap; not surfaced standalone
    "has_prior_auth": {},
    "referral_required": {},
    "referral_present": {},
}

NUMERIC_PHRASES = {
    "days_to_submit": "the claim was submitted {value:.0f} days after service (later submissions carry more denial risk)",
    "payment_ratio": "expected payment is only {value:.0%} of the billed amount",
    "num_procedures": "the claim lists {value:.0f} procedure codes",
    "num_diagnoses": "the claim lists {value:.0f} diagnosis codes",
    "total_billed": "the total billed amount is ${value:,.0f}",
    "expected_payment": "the expected payment is ${value:,.0f}",
}


def _base_feature_and_category(encoded_name: str):
    """Split a one-hot-encoded name like 'payer_type_Commercial' into
    ('payer_type', 'Commercial'). Returns (encoded_name, None) if not one-hot.
    """
    for base in ["payer_id", "payer_type", "visit_type"]:
        prefix = base + "_"
        if encoded_name.startswith(prefix):
            return base, encoded_name[len(prefix):]
    return encoded_name, None


def describe_feature(encoded_name: str, raw_row: dict) -> str:
    """Return a plain-English phrase for one feature, or None if not surfaceable."""
    base, category = _base_feature_and_category(encoded_name)

    if category is not None:
        if base == "payer_id":
            return None  # payer_id alone isn't an actionable phrase; payer_type is
        if base == "payer_type":
            return f"the payer type is {category}"
        if base == "visit_type":
            article = "an" if category.lower().startswith(("i", "o", "e")) else "a"
            return f"this is {article} {category.lower()} visit"
        return None

    if encoded_name in BINARY_PHRASES:
        raw_value = raw_row.get(encoded_name)
        try:
            raw_value = int(raw_value)
        except (TypeError, ValueError):
            return None
        return BINARY_PHRASES[encoded_name].get(raw_value)

    if encoded_name in NUMERIC_PHRASES:
        raw_value = raw_row.get(encoded_name)
        if raw_value is None:
            return None
        try:
            return NUMERIC_PHRASES[encoded_name].format(value=float(raw_value))
        except (TypeError, ValueError):
            return None

    return None


def top_risk_factors_for_row(
    feature_names: list, shap_row: np.ndarray, raw_row: dict, top_n: int = 3
) -> list:
    """Return up to `top_n` human-readable phrases for the features that pushed
    this row's score toward denial the most (positive SHAP contributions only --
    negative contributions push toward "not denied" and aren't a risk factor).
    """
    candidates = []
    for name, shap_val in zip(feature_names, shap_row):
        if shap_val <= 0:
            continue
        phrase = describe_feature(name, raw_row)
        if phrase:
            candidates.append((shap_val, phrase))

    candidates.sort(key=lambda x: -x[0])
    seen = set()
    result = []
    for _, phrase in candidates:
        if phrase in seen:
            continue
        seen.add(phrase)
        result.append(phrase)
        if len(result) >= top_n:
            break
    return result
