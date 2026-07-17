"""Metric helpers centered on the review team's 25%-of-claims capacity constraint."""

import numpy as np
import pandas as pd


def threshold_at_percentile(scores: np.ndarray, percentile: float) -> float:
    """Probability cutoff such that `percentile` fraction of `scores` fall above it.

    e.g. percentile=0.75 -> the cutoff that flags the top 25% of claims as high risk.
    """
    return float(np.quantile(scores, 1 - percentile))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    """Fraction of actual denials captured if only the top k_fraction of claims
    (ranked by predicted score) are reviewed. This is the "denial capture rate"
    referenced throughout the write-up and README.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_score)
    k = max(1, int(np.ceil(n * k_fraction)))
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    total_positives = y_true.sum()
    if total_positives == 0:
        return float("nan")
    return float(y_true[top_k_idx].sum() / total_positives)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    """Of the claims flagged for review (top k_fraction), what share were actually denied."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_score)
    k = max(1, int(np.ceil(n * k_fraction)))
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    return float(y_true[top_k_idx].mean())


def capture_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    """Cumulative % of denials captured vs % of claims reviewed, for plotting."""
    fractions = np.linspace(0.01, 1.0, n_points)
    capture = [recall_at_k(y_true, y_score, f) for f in fractions]
    return pd.DataFrame({"fraction_reviewed": fractions, "fraction_denials_captured": capture})


def recall_at_k_table(y_true: np.ndarray, y_score: np.ndarray, k_fractions=None) -> pd.DataFrame:
    """For each review-capacity fraction k, the threshold that produces exactly
    that top-k cutoff, plus recall/precision at that point.

    This is the direct answer to "what threshold suits recall@25% best": the
    review team's capacity (25%) determines k, and k determines the threshold
    -- there's no separate free choice of threshold once k is fixed. This
    table shows how recall/precision would change if the review team's
    capacity were slightly larger or smaller, i.e. how sensitive the chosen
    operating point is to that one constraint.
    """
    if k_fractions is None:
        k_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rows = []
    for k in k_fractions:
        rows.append(
            {
                "review_capacity": k,
                "threshold": threshold_at_percentile(y_score, k),
                "recall_at_k": recall_at_k(y_true, y_score, k),
                "precision_at_k": precision_at_k(y_true, y_score, k),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray, thresholds=None) -> pd.DataFrame:
    """Accuracy/precision/recall/F1 at each threshold in `thresholds`, for one model.

    Used to answer "at what threshold is each model most accurate" -- accuracy alone
    is not the model-selection metric (see recall_at_k / the 25% review constraint),
    but it's reported here for a full threshold-by-threshold comparison.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "flagged_fraction": float(y_pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def best_accuracy_threshold(y_true: np.ndarray, y_score: np.ndarray, thresholds=None) -> dict:
    """The threshold (from a fine grid) that maximizes accuracy for this model's scores."""
    from sklearn.metrics import accuracy_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)

    best = {"threshold": 0.5, "accuracy": -1.0}
    for t in thresholds:
        acc = accuracy_score(y_true, (y_score >= t).astype(int))
        if acc > best["accuracy"]:
            best = {"threshold": float(t), "accuracy": float(acc)}
    return best


def classification_summary(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Bundle of headline metrics reported in evaluate.py and the write-up."""
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_25pct": recall_at_k(y_true, y_score, 0.25),
        "precision_at_25pct": precision_at_k(y_true, y_score, 0.25),
        "threshold": float(threshold),
        "accuracy_at_threshold": float((tp + tn) / (tp + tn + fp + fn)),
        "precision_at_threshold": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        "recall_at_threshold": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
