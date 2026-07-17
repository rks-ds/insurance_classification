"""Evaluate a saved model on a given split of claims_history.csv.

Usage:
    python src/evaluate.py --model_path outputs/model.pkl --data_path data/claims_history.csv --split test
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_feature_matrix, load_history
from src.metrics import best_accuracy_threshold, classification_summary, recall_at_k_table, threshold_sweep
from src.plots import (
    plot_capture_curve,
    plot_confusion_matrices_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_pr,
    plot_pr_comparison,
    plot_recall_at_k_table,
    plot_roc,
    plot_shap_summary,
    plot_threshold_accuracy,
)
from src.shap_utils import compute_shap_values, mean_abs_shap

DISPLAY_NAME = {"logistic_regression": "Logistic Regression", "lightgbm": "LightGBM"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/model.pkl")
    parser.add_argument("--data_path", default="data/claims_history.csv")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--figures_dir", default="outputs/figures")
    parser.add_argument("--metrics_out", default="outputs/metrics_test.json")
    parser.add_argument("--threshold_sweep_out", default="outputs/threshold_sweep.csv")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    threshold = artifact["high_threshold"]

    df = load_history(args.data_path)
    subset = df[df["split"] == args.split].reset_index(drop=True)
    X = preprocessor.transform(get_feature_matrix(subset))
    y_true = subset["is_denied"].values
    y_score = model.predict_proba(X)[:, 1]

    summary = classification_summary(y_true, y_score, threshold)
    summary["model_name"] = artifact["model_name"]
    summary["split"] = args.split

    print(f"\n--- Evaluation on '{args.split}' split ({summary['n']} claims, "
          f"{summary['n_positive']} actual denials) ---")
    print(f"Model: {summary['model_name']}")
    print(f"ROC-AUC: {summary['roc_auc']:.3f}")
    print(f"PR-AUC:  {summary['pr_auc']:.3f}")
    print(f"Recall @ top 25% (denial capture rate): {summary['recall_at_25pct']:.3f}")
    print(f"Precision @ top 25%: {summary['precision_at_25pct']:.3f}")
    print(f"At fixed high-risk threshold ({threshold:.4f}): "
          f"accuracy={summary['accuracy_at_threshold']:.3f}, "
          f"precision={summary['precision_at_threshold']:.3f}, "
          f"recall={summary['recall_at_threshold']:.3f}")
    print(f"Confusion matrix: {summary['confusion_matrix']}")

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    plot_capture_curve(y_true, y_score, f"{args.figures_dir}/capture_curve.png")
    plot_roc(y_true, y_score, f"{args.figures_dir}/roc_curve.png")
    plot_pr(y_true, y_score, f"{args.figures_dir}/pr_curve.png")
    plot_confusion_matrix(summary["confusion_matrix"], f"{args.figures_dir}/confusion_matrix.png")

    from src.data import get_feature_names

    feature_names = get_feature_names(preprocessor)
    shap_values = compute_shap_values(artifact, X)
    importances = mean_abs_shap(shap_values)
    plot_feature_importance(feature_names, importances, f"{args.figures_dir}/feature_importance.png")
    plot_shap_summary(
        shap_values, X, feature_names, f"{args.figures_dir}/shap_summary.png",
        title=f"SHAP Summary -- {DISPLAY_NAME.get(artifact['model_name'], artifact['model_name'])} ({args.split} set)",
    )

    baseline_recall = artifact.get("baseline_val_recall_at_25pct")
    main_recall = artifact.get("main_val_recall_at_25pct")
    if baseline_recall is not None and main_recall is not None:
        plot_model_comparison(
            ["Logistic Regression\n(baseline)", "LightGBM\n(final model)"],
            [baseline_recall, main_recall],
            f"{args.figures_dir}/model_comparison.png",
        )

    # --- Per-model results analysis: PR curve overlay, confusion matrices side by
    # side, and a threshold -> accuracy sweep for every model that was trained,
    # not just the winner. Each model is scored at ITS OWN validation-derived
    # top-25% threshold, so this is an apples-to-apples comparison at a
    # consistent operating point rather than one model's tuned threshold applied
    # to both.
    all_models = artifact.get("all_models")
    per_model_threshold = artifact.get("per_model_high_threshold", {})
    threshold_sweep_rows = []
    model_comparison = {}
    if all_models:
        pr_curves = {}
        cms = {}
        acc_sweeps = {}
        for name, m in all_models.items():
            display = DISPLAY_NAME.get(name, name)
            m_score = m.predict_proba(X)[:, 1]
            pr_curves[display] = (y_true, m_score)

            m_threshold = per_model_threshold.get(name, threshold)
            m_summary = classification_summary(y_true, m_score, m_threshold)
            cms[f"{display}\n(t={m_threshold:.2f})"] = m_summary["confusion_matrix"]

            sweep_df = threshold_sweep(y_true, m_score)
            sweep_df.insert(0, "model", display)
            threshold_sweep_rows.append(sweep_df)
            best = best_accuracy_threshold(y_true, m_score)
            acc_sweeps[display] = (sweep_df, best)

            model_comparison[name] = {
                "display_name": display,
                "pr_auc": m_summary["pr_auc"],
                "top25pct_threshold": m_threshold,
                "at_top25pct_threshold": {
                    "accuracy": m_summary["accuracy_at_threshold"],
                    "precision": m_summary["precision_at_threshold"],
                    "recall": m_summary["recall_at_threshold"],
                    "confusion_matrix": m_summary["confusion_matrix"],
                },
                "best_accuracy_threshold": best["threshold"],
                "best_accuracy": best["accuracy"],
            }

            print(f"\n[{display}] best-accuracy threshold={best['threshold']:.2f} "
                  f"-> accuracy={best['accuracy']:.3f}; "
                  f"at top-25% threshold t={m_threshold:.3f} -> "
                  f"accuracy={m_summary['accuracy_at_threshold']:.3f}, "
                  f"precision={m_summary['precision_at_threshold']:.3f}, "
                  f"recall={m_summary['recall_at_threshold']:.3f}, "
                  f"confusion_matrix={m_summary['confusion_matrix']}")

        plot_pr_comparison(pr_curves, f"{args.figures_dir}/pr_curve_comparison.png")
        plot_confusion_matrices_comparison(cms, f"{args.figures_dir}/confusion_matrix_comparison.png")
        plot_threshold_accuracy(acc_sweeps, f"{args.figures_dir}/threshold_accuracy.png")

        import pandas as pd

        full_sweep = pd.concat(threshold_sweep_rows, ignore_index=True)
        Path(args.threshold_sweep_out).parent.mkdir(parents=True, exist_ok=True)
        full_sweep.to_csv(args.threshold_sweep_out, index=False)
        print(f"\nSaved per-model threshold/accuracy sweep to {args.threshold_sweep_out}")

        with open(Path(args.metrics_out).parent / "model_comparison.json", "w") as f:
            json.dump(model_comparison, f, indent=2)
        print(f"Saved per-model comparison summary to {Path(args.metrics_out).parent / 'model_comparison.json'}")

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {args.metrics_out}")
    print(f"Saved figures to {args.figures_dir}/")


if __name__ == "__main__":
    main()
