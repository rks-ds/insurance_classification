"""Train baseline and main denial-risk models, select the winner by
recall@25% on the validation split, and persist the winner + fitted
preprocessor + operating thresholds to a single pickle.

Usage:
    python src/train.py --data_path data/claims_history.csv --seed 42
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_preprocessor, get_feature_matrix, load_history
from src.metrics import recall_at_k, threshold_at_percentile

REVIEW_CAPACITY = 0.25  # review team can inspect the top 25% of claims by risk score


def split_by_column(df, split_col="split"):
    return {name: df[df[split_col] == name].reset_index(drop=True) for name in ["train", "validation", "test"]}


# Small grid, deliberately biased toward shallow/regularized configs: with
# ~2,100 training rows, an earlier default (max_depth=4, n_estimators=200,
# no min_child_samples floor) scored below the linear baseline on validation
# recall@25% -- a clear overfitting signal, which is why every config here
# stays shallow (max_depth<=3, num_leaves<=8) rather than searching deeper.
# See outputs/lightgbm_grid_search.csv for the full sweep after each run.
LIGHTGBM_GRID = [
    {"max_depth": md, "num_leaves": nl, "min_child_samples": mcs, "n_estimators": ne, "learning_rate": lr}
    for md in [2, 3]
    for nl in [4, 8]
    for mcs in [30, 50]
    for ne in [50, 100]
    for lr in [0.05]
]


def tune_lightgbm(X_train, y_train, X_val, y_val, seed: int, k_fraction: float):
    """Grid search LightGBM hyperparameters, selecting on validation recall@k
    (the same metric used for final model selection) rather than accuracy or
    AUC, so the tuning objective matches what the model is actually chosen for.
    """
    best_model = None
    best_params = None
    best_score = -1.0
    results = []

    for params in LIGHTGBM_GRID:
        model = LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        val_score = model.predict_proba(X_val)[:, 1]
        recall = recall_at_k(y_val, val_score, k_fraction)
        results.append({**params, "val_recall_at_k": recall})
        if recall > best_score:
            best_score = recall
            best_params = params
            best_model = model

    return best_model, best_params, best_score, results


def train_models(X_train, y_train, X_val, y_val, seed: int, k_fraction: float):
    baseline = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed
    )
    baseline.fit(X_train, y_train)

    main_model, best_params, best_val_recall, grid_results = tune_lightgbm(
        X_train, y_train, X_val, y_val, seed, k_fraction
    )
    print(f"LightGBM grid search: {len(grid_results)} configs tried, "
          f"best val recall@{k_fraction:.0%} = {best_val_recall:.3f} at {best_params}")

    return {"logistic_regression": baseline, "lightgbm": main_model}, grid_results, best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/claims_history.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/model.pkl")
    args = parser.parse_args()

    np.random.seed(args.seed)

    df = load_history(args.data_path)
    splits = split_by_column(df)

    preprocessor = build_preprocessor()
    X_train_raw = get_feature_matrix(splits["train"])
    X_train = preprocessor.fit_transform(X_train_raw)
    y_train = splits["train"]["is_denied"].values

    X_val = preprocessor.transform(get_feature_matrix(splits["validation"]))
    y_val = splits["validation"]["is_denied"].values

    models, lightgbm_grid_results, lightgbm_best_params = train_models(
        X_train, y_train, X_val, y_val, args.seed, REVIEW_CAPACITY
    )

    print(f"{'Model':<20} {'Val ROC-AUC':<12} {'Val recall@25%':<16}")
    scores = {}
    for name, model in models.items():
        val_score = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, val_score)
        recall25 = recall_at_k(y_val, val_score, REVIEW_CAPACITY)
        scores[name] = recall25
        print(f"{name:<20} {auc:<12.3f} {recall25:<16.3f}")

    winner_name = max(scores, key=scores.get)
    winner = models[winner_name]
    print(f"\nSelected model: {winner_name} (highest val recall@25%)")

    val_scores_winner = winner.predict_proba(X_val)[:, 1]
    high_threshold = threshold_at_percentile(val_scores_winner, REVIEW_CAPACITY)
    medium_threshold = threshold_at_percentile(val_scores_winner, 0.50)
    print(f"High-risk threshold (top 25% cutoff, from validation): {high_threshold:.4f}")
    print(f"Medium-risk threshold (top 50% cutoff, from validation): {medium_threshold:.4f}")

    # Each model gets its own top-25% threshold (derived from its own validation
    # score distribution) so evaluate.py can fairly compare baseline vs. final
    # model at the same operating point, even though only `winner` is used for
    # scoring current_claims.csv.
    per_model_high_threshold = {
        name: threshold_at_percentile(model.predict_proba(X_val)[:, 1], REVIEW_CAPACITY)
        for name, model in models.items()
    }

    # Background sample for shap.LinearExplainer (logistic regression): a fixed
    # subsample of the TRAINING distribution, so SHAP values for any later batch
    # (test set, current_claims.csv) are always "vs. what the model was fit on,"
    # not "vs. whatever else happens to be in that batch."
    rng = np.random.RandomState(args.seed)
    bg_idx = rng.choice(X_train.shape[0], size=min(200, X_train.shape[0]), replace=False)
    shap_background = X_train[bg_idx]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": winner_name,
        "model": winner,
        "all_models": models,
        "per_model_high_threshold": per_model_high_threshold,
        "preprocessor": preprocessor,
        "high_threshold": high_threshold,
        "medium_threshold": medium_threshold,
        "review_capacity": REVIEW_CAPACITY,
        "seed": args.seed,
        "baseline_val_recall_at_25pct": scores.get("logistic_regression"),
        "main_val_recall_at_25pct": scores.get("lightgbm"),
        "lightgbm_best_params": lightgbm_best_params,
        "shap_background": shap_background,
    }
    with open(args.out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model artifact to {args.out}")

    import pandas as pd

    grid_path = Path(args.out).parent / "lightgbm_grid_search.csv"
    pd.DataFrame(lightgbm_grid_results).sort_values("val_recall_at_k", ascending=False).to_csv(
        grid_path, index=False
    )
    print(f"Saved LightGBM grid search results to {grid_path}")


if __name__ == "__main__":
    main()
