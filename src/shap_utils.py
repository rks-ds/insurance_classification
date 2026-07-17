"""Shared SHAP computation so evaluate.py and score.py explain models the same
way, instead of each using a different ad hoc importance proxy.

LightGBM uses shap.TreeExplainer (exact, fast). Logistic Regression uses
shap.LinearExplainer against a background sample of the TRAINING distribution
(saved in the model artifact by train.py) -- using the scoring batch itself as
its own background would bias each row's SHAP values toward "how does this
claim compare to the other claims in today's batch" rather than "how does it
compare to the claims the model was fit on."
"""

import numpy as np
import shap


def compute_shap_values(artifact: dict, X: np.ndarray) -> np.ndarray:
    """Return a (n_rows, n_features) array of SHAP values for whichever model
    is stored as `artifact["model"]`, using `artifact["model_name"]` to pick
    the right explainer.
    """
    model = artifact["model"]
    model_name = artifact["model_name"]
    return compute_shap_values_for(model_name, model, X, artifact.get("shap_background"))


def compute_shap_values_for(model_name: str, model, X: np.ndarray, background: np.ndarray = None) -> np.ndarray:
    if model_name == "lightgbm":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # SHAP's binary-classifier TreeExplainer output shape has changed across
        # versions: older releases return a list of two (n_rows, n_features)
        # arrays (one per class); some return a single 3D (n_rows, n_features, 2)
        # array; the currently-pinned version returns a single 2D array already
        # in the positive-class margin space. Normalize all three to 2D so
        # downstream code (mean_abs_shap, risk_factors) never has to know which.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class-1 (denied) contributions
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values

    if model_name == "logistic_regression":
        bg = background if background is not None else X
        explainer = shap.LinearExplainer(model, bg)
        shap_values = explainer.shap_values(X)
        return shap_values

    raise ValueError(f"No SHAP explainer configured for model_name={model_name!r}")


def mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    """Global feature importance: mean |SHAP value| per feature, across rows."""
    return np.abs(shap_values).mean(axis=0)
