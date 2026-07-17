"""Data loading and feature engineering for the claims denial model."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "is_denied"

# Columns that must never reach the model: identifiers, leakage-prone fields,
# and metadata used only for splitting/bookkeeping.
DROP_COLS = ["claim_id", "split", "service_month", "denial_reason"]

CATEGORICAL_COLS = ["payer_id", "payer_type", "visit_type"]

BINARY_COLS = [
    "prior_auth_required",
    "has_prior_auth",
    "is_in_network",
    "missing_documentation_flag",
    "eligibility_verified",
    "referral_required",
    "referral_present",
    "auth_gap",
    "referral_gap",
]

NUMERIC_COLS = [
    "total_billed",
    "expected_payment",
    "num_procedures",
    "num_diagnoses",
    "days_to_submit",
    "payment_ratio",
]

FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS


def load_history(path: str) -> pd.DataFrame:
    """Load claims_history.csv, keeping split/target columns for downstream use."""
    df = pd.read_csv(path)
    return engineer_features(df)


def load_current(path: str) -> pd.DataFrame:
    """Load current_claims.csv (no split/target columns present)."""
    df = pd.read_csv(path)
    return engineer_features(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived signal columns used by both history and current claims.

    auth_gap / referral_gap flag claims where an approval is required but not
    on file -- these showed the strongest lift toward denial in exploration
    (auth_gap: ~46% denial rate vs ~22% baseline).
    """
    df = df.copy()
    df["auth_gap"] = (
        (df["prior_auth_required"] == 1) & (df["has_prior_auth"] == 0)
    ).astype(int)
    df["referral_gap"] = (
        (df["referral_required"] == 1) & (df["referral_present"] == 0)
    ).astype(int)
    df["payment_ratio"] = df["expected_payment"] / df["total_billed"].replace(0, 1)
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only model-input columns, in a fixed order."""
    return df[FEATURE_COLS].copy()


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer shared by both models.

    handle_unknown='ignore' on the one-hot encoder is a defensive choice: every
    payer_id/payer_type/visit_type value in current_claims.csv does appear in
    claims_history.csv, but scoring code shouldn't rely on that staying true --
    a genuinely new payer showing up later should degrade gracefully rather
    than raise at inference time.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS + BINARY_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Human-readable feature names after the ColumnTransformer has been fit."""
    num_names = NUMERIC_COLS + BINARY_COLS
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    return num_names + cat_names
