# Five ML habits, put to the test on 89 imbalanced datasets

An empirical study that takes five common machine-learning habits and checks whether they
actually hold up, on **89 imbalanced binary datasets** (imblearn + OpenML), using
**Average Precision** (PR-AUC) and a **fixed, shared test set**. Everything is compared with
paired t-tests on the same splits, and — crucially — **at equal wall-clock time**, not at an
equal number of models tried.

It's written up as a four-part series:

1. **Validation splits & retraining** — do you need a separate validation set to pick the
   model on top of the one cross-validation already uses for early stopping? (No. And
   retraining only helps if you held data out.)
2. **Hyperparameter search** — grid vs random, TPE, CMA-ES, GP-Bayesian, Hyperband and BOHB.
   Nobody beats a sensible grid on accuracy; multi-fidelity (Hyperband/BOHB) matches it in
   ~15× less time.
3. **Ensembles** — "a LightGBM is faster than an ensemble" is true and irrelevant: at equal
   time, a greedy blend evaluated out-of-fold beats "more configs of a single model."
4. **Speculative feature engineering** — trees mostly ignore random features, but the habit
   still costs training time for zero gain: a strictly worse use of compute.

## Method (shared across all parts)

- 89 imbalanced binary datasets; metric = Average Precision.
- Fixed 30% test set (capped at 6,000 rows), touched once.
- Fraction sweep of each dataset's development pool (10%…100%) so every dataset is present
  at every point on the curve (avoids composition artifacts).
- Base learner: LightGBM, number of trees set by early stopping; per-fold preprocessing,
  no leakage.
- Every p-value is a paired t-test on the same splits.

## Layout

- `core.py`, `kfold_core` helpers — datasets, preprocessing, CV, scoring.
- `hpo_study.py`, `hpo_mf.py` — hyperparameter-search experiments (full-budget + multi-fidelity).
- `isotime_*.py`, `analyze_*.py`, `make_fig_A.py` — ensemble / equal-time experiments and figures.
- `feature_eng.py`, `analyze_feateng.py` — speculative feature engineering.
- `results/` — raw per-cell JSON results (the numbers behind every claim).
- `summary_*.json` / `summary_*.csv` — aggregated summaries used by the write-ups.
- `make_tables.py`, `build_article_html*.py`, `medium_html.py` — figures, colored tables and
  article builders.
- `medium/articles/` — the four-part series (Markdown source).

Figures are generated in English with `FIGLANG=en`.
