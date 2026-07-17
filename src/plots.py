"""Plotting helpers. All functions save a PNG to `out_path` and return that path."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.metrics import capture_curve


def plot_capture_curve(y_true, y_score, out_path: str, highlight_fraction: float = 0.25):
    df = capture_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df["fraction_reviewed"] * 100, df["fraction_denials_captured"] * 100,
            label="Model", color="#1f5fa8", linewidth=2)
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random review order")
    from src.metrics import recall_at_k
    y_at_cut = recall_at_k(y_true, y_score, highlight_fraction) * 100
    ax.axvline(highlight_fraction * 100, color="#c0392b", linestyle=":", linewidth=1.5)
    ax.scatter([highlight_fraction * 100], [y_at_cut], color="#c0392b", zorder=5)
    ax.annotate(f"{y_at_cut:.0f}% of denials\ncaptured at {highlight_fraction*100:.0f}% reviewed",
                xy=(highlight_fraction * 100, y_at_cut), xytext=(highlight_fraction * 100 + 8, y_at_cut - 15),
                fontsize=9, color="#c0392b")
    ax.set_xlabel("% of claims reviewed (ranked by risk score)")
    ax.set_ylabel("% of actual denials captured")
    ax.set_title("Denial Capture Rate vs. Review Capacity")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc(y_true, y_score, out_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#1f5fa8", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr(y_true, y_score, out_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, color="#1f5fa8", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm: dict, out_path: str):
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Not Denied", "Predicted: Denied"])
    ax.set_yticklabels(["Actual: Not Denied", "Actual: Denied"])
    ax.set_title("Confusion Matrix at Chosen Threshold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance(feature_names, importances, out_path: str, top_n: int = 12):
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    vals = [importances[i] for i in order][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names, vals, color="#1f5fa8")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_recall_at_k_table(df, out_path: str, highlight_fraction: float = 0.25):
    """Recall@k and precision@k vs. review-capacity fraction, with the actual
    operating point (25%) marked -- shows how the metric and its implied
    threshold move if review capacity were slightly different.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df["review_capacity"] * 100, df["recall_at_k"] * 100, marker="o",
            color="#1f5fa8", linewidth=2, label="Recall@k (denial capture rate)")
    ax.plot(df["review_capacity"] * 100, df["precision_at_k"] * 100, marker="o",
            color="#c0392b", linewidth=2, label="Precision@k")
    row = df.iloc[(df["review_capacity"] - highlight_fraction).abs().idxmin()]
    ax.axvline(highlight_fraction * 100, color="gray", linestyle=":", linewidth=1.5)
    ax.annotate(f"t={row['threshold']:.2f}", xy=(highlight_fraction * 100, 5),
                fontsize=8, color="gray", ha="center")
    ax.set_xlabel("Review capacity (% of claims reviewed)")
    ax.set_ylabel("%")
    ax.set_title("Recall@k / Precision@k vs. Review Capacity")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr_comparison(curves: dict, out_path: str):
    """PR curves for multiple models on the same axes.

    `curves` maps display name -> (y_true, y_score).
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = ["#1f5fa8", "#c0392b", "#27ae60"]
    for (name, (y_true, y_score)), color in zip(curves.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{name} (PR-AUC={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve: Baseline vs. Final Model")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrices_comparison(cms: dict, out_path: str):
    """Side-by-side confusion matrices for multiple models.

    `cms` maps display name -> confusion_matrix dict {tn, fp, fn, tp}.
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Not Denied", "Pred: Denied"], fontsize=8)
        ax.set_yticklabels(["Actual: Not Denied", "Actual: Denied"], fontsize=8)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion Matrix at Each Model's Top-25% Threshold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_threshold_accuracy(sweeps: dict, out_path: str):
    """Accuracy vs. threshold for multiple models on the same axes, with each
    model's own best-accuracy point marked.

    `sweeps` maps display name -> (threshold_sweep_df, best_threshold_dict).
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#1f5fa8", "#c0392b", "#27ae60"]
    for i, ((name, (df, best)), color) in enumerate(zip(sweeps.items(), colors)):
        ax.plot(df["threshold"], df["accuracy"] * 100, color=color, linewidth=2, label=name)
        ax.scatter([best["threshold"]], [best["accuracy"] * 100], color=color, zorder=5)
        # Stack annotations vertically (rather than at a fixed offset) so two
        # models peaking at a similar threshold/accuracy don't overlap.
        y_text = best["accuracy"] * 100 - 8 - 8 * i
        ax.annotate(f"{name}: {best['accuracy']*100:.1f}% @ t={best['threshold']:.2f}",
                     xy=(best["threshold"], best["accuracy"] * 100),
                     xytext=(min(best["threshold"] + 0.03, 0.65), y_text),
                     fontsize=8, color=color)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Threshold, by Model")
    ax.legend(loc="lower center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_shap_summary(shap_values, X, feature_names, out_path: str, max_display: int = 12, title: str = None):
    """SHAP beeswarm plot: per-feature distribution of each row's SHAP value,
    colored by that row's (scaled) feature value. Complements the plain mean-
    |SHAP| bar chart (plot_feature_importance) by showing DIRECTION (does a
    high value push risk up or down) and spread, not just overall magnitude.
    """
    import shap

    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, max_display=max_display, show=False
    )
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(model_names, recalls, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(model_names, [r * 100 for r in recalls], color=["#95a5a6", "#1f5fa8"])
    ax.set_ylabel("Recall @ top 25% (%)")
    ax.set_title("Denial Capture Rate: Baseline vs. Final Model")
    for bar, r in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{r*100:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
