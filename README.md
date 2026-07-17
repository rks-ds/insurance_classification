# Claims Denial Risk Model

Predicts whether a hospital claim will be denied, ranks `current_claims.csv`
by risk so a review team that can only inspect the **top 25%** of claims
focuses on the ones most likely to be denied, and generates plain-English,
analyst-facing explanations for the 10 highest-risk current claims.

## Approach, in short

- **Metric**: model selection and headline reporting are built around
  **recall @ top 25%** ("denial capture rate") — the fraction of actual
  denials that would be caught if the review team only ever inspects the
  top quartile of claims by risk score. This mirrors the stated review
  capacity constraint directly, instead of optimizing plain accuracy or AUC.
- **Models compared**: a `LogisticRegression` baseline vs. a `LightGBM`
  classifier, both trained on `train` and model-selected on `validation` by
  recall@25%, then reported once on `test`. LightGBM's hyperparameters
  (`max_depth`, `num_leaves`, `min_child_samples`, `n_estimators`) are grid
  searched in `src/train.py`, selecting on validation recall@25% -- the same
  metric used for the final baseline-vs-LightGBM choice -- and the full sweep
  is saved to `outputs/lightgbm_grid_search.csv` on every run. On this dataset
  size (~2,100 training rows) the best LightGBM config and Logistic Regression
  came out statistically tied (val recall@25% ≈ 0.495 both); logistic
  regression was kept as the final model for its interpretability and
  stability.
- **Risk tiers / thresholds**: `High` / `Medium` / `Low` cutoffs are fixed
  probability values derived once from the **validation** set's score
  distribution (75th and 50th percentiles) and then applied unchanged to
  the test set and to `current_claims.csv`. This keeps the operating point
  stable across batches instead of re-deriving a threshold from whatever
  claims happen to be scored that day.
  - High: `denial_probability >= 0.5942` (top ~25% on validation)
  - Medium: `0.4182 <= denial_probability < 0.5942`
  - Low: `denial_probability < 0.4182`
- **Per-claim risk factors**: SHAP values from the model are translated into
  plain-English phrases (e.g. "prior authorization is required but not on
  file") — these are what both `top_risk_factors` and the LLM prompt are
  grounded in, so nothing shown to the analyst is invented.
- **LLM explanations**: the top 10 highest-risk current claims get a 2-3
  sentence explanation generated with an LLM (OpenAI `gpt-4o-mini`), grounded
  only in the claim's fields and its SHAP-derived risk factors. If no working
  API key/quota is available at runtime, `explain.py` automatically falls
  back to manually drafted explanations built from the same risk-factor data
  (see "LLM access" below — this fallback is what actually ran for this
  submission).

## Repository layout

```
data/                        claims_history.csv, current_claims.csv
notebooks/
  eda.ipynb                  univariate + bivariate exploratory analysis, with an
                             inference note after every chart
src/
  data.py                    loading + feature engineering + preprocessing pipeline
  metrics.py                 recall@k / capture-rate / threshold-sweep / classification summary helpers
  plots.py                   capture curve, ROC, PR, confusion matrix, feature importance, SHAP summary, model comparison
  shap_utils.py               shared SHAP computation (LinearExplainer / TreeExplainer) for evaluate.py + score.py
  risk_factors.py            SHAP values -> plain-English risk-factor phrases
  train.py                   grid-searches LightGBM, trains baseline + main model, model-selects, saves outputs/model.pkl
  evaluate.py                evaluates a saved model on a chosen split, writes metrics + figures
  score.py                   scores current_claims.csv -> predictions_current_claims.csv
  explain.py                 LLM explanation generation for the top 10 highest-risk claims
outputs/
  model.pkl                  trained model + preprocessor + thresholds + SHAP background sample
  lightgbm_grid_search.csv   every LightGBM hyperparameter config tried, ranked by val recall@25%
  metrics_test.json          test-set metrics for the selected model
  model_comparison.json      the same metrics computed for both models, at each model's own threshold
  threshold_sweep.csv        accuracy/precision/recall/F1 at each threshold, for both models
  figures/                   all generated plots (PNG)
  explanations_top10.json    full prompt + input + output record per explained claim
predictions_current_claims.csv   final scored output, sorted by denial_probability desc
writeup/
  build_writeup.py           assembles the PDF write-up from figures + metrics
  writeup.pdf                final write-up deliverable
```

## How to run

```bash
pip install -r requirements.txt

python src/train.py --data_path data/claims_history.csv --seed 42
python src/evaluate.py --model_path outputs/model.pkl --data_path data/claims_history.csv --split test
python src/score.py --model_path outputs/model.pkl --data_path data/current_claims.csv --out predictions_current_claims.csv
python src/explain.py --predictions predictions_current_claims.csv --data_path data/current_claims.csv --top_n 10
python writeup/build_writeup.py
```

Run all commands from the repository root.

## Exploratory data analysis

`notebooks/eda.ipynb` is a standalone univariate + bivariate analysis of
`data/claims_history.csv`, independent of the modeling pipeline above. Every
chart is followed by a short **Inference** note explaining what it shows and
why it matters -- these notes are the source of the data findings summarized
in `writeup/writeup.pdf` and the feature choices in `src/data.py`.

Open it interactively:

```bash
jupyter notebook notebooks/eda.ipynb
```

Or re-run it headless and save outputs in place:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

## LLM access

`src/explain.py` uses the `openai` Python package and reads `OPENAI_API_KEY`
from the environment. In this environment the key present did not have an
active quota, so the script's built-in fallback path ran instead: it drafts
each explanation from the same grounded risk-factor data the LLM prompt would
have received, using a small per-risk-factor action map (see
`ACTION_BY_RISK_FACTOR` in `src/explain.py`) rather than one generic template.
The exact prompt template that would be sent to the LLM is documented in
`src/explain.py` (`SYSTEM_PROMPT` / `USER_PROMPT_TEMPLATE`) and in the PDF
write-up. Dropping in a funded `OPENAI_API_KEY` and re-running the same
command requires no code changes.

## Notes on the data

- `claim_id`, `split`, `service_month`, and `denial_reason` are excluded from
  model inputs (`denial_reason` only exists after a denial and would leak the
  target; `service_month`'s only two current-claims values, 2025-01/02, don't
  overlap with the 2024 training months, so it isn't a usable feature).
- Every `payer_id`, `payer_type`, and `visit_type` value in `current_claims.csv`
  also appears in `claims_history.csv` -- no genuinely unseen categories in
  this dataset. The preprocessing pipeline still one-hot encodes with
  `handle_unknown="ignore"` (see `src/data.py:build_preprocessor`) as a
  defensive default, so scoring wouldn't fail if a new payer ever did show up.
