# Don't select on precision@k: model selection for fixed-budget top-k problems

Code and data pipeline for the study **"Precision@k is a great target and a dangerous judge:
choosing your model with a metric other than your business objective"**.

Across **93 imbalanced datasets** (90 real from `imbalanced-learn` + OpenML, 3 synthetic),
sweeping the inspection budget *K*, we show that:

1. **Precision@k cannot be a training loss** (piecewise-constant, zero gradient) — worked example included.
2. As a **model-selection** metric it is *significantly worse* than **average precision** —
   in **all three model families** tested (gradient boosting, random forest, logistic regression),
   paired Wilcoxon p ≤ 1.2e-3: selecting on the exact metric you care about yields a *worse*
   held-out precision@k than selecting on a more stable proxy — the winner's curse. (Log-loss
   is tied-best for gradient boosting but its edge is model-dependent; AUC is never better.)
3. What governs selection difficulty is **not a ratio** (n_pos/K) but the **absolute number of true
   positives inside the budget** (≈ precision × K); its model-free proxy `min(K, n_pos)` is knowable a priori.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python experiment_real3.py     # trains the model banks, sweeps K -> results_real3.json  (~50 min, downloads datasets on first run)
python meta_real3.py           # aggregates + paired Wilcoxon tests + figures figT0/figT2/figT5
python fig_apriori.py          # figure figT6: a-priori min(K, n_pos) vs endogenous precision x K
python fig_step.py             # figure 1: precision@k step function vs smooth log-loss
python fig_gradient.py         # figure 2: one boosting step, log-loss gradient vs precision@k
```

The paper (LaTeX) is in `paper/` and reuses the generated `.png` figures (copied to `paper/figs/`).

Everything runs from a fixed seed. Datasets are downloaded automatically (imbalanced-learn +
OpenML) and cached under `data_cache/` on first run.

## Layout

| file | what it is |
|---|---|
| `experiment_real3.py` | main experiment: train 40 LightGBM configs per dataset, sweep budget K, bootstrap model selection, record normalized regret per metric |
| `meta_real3.py` | aggregation, paired Wilcoxon tests, and result figures |
| `fig_apriori.py` | model-free `min(K, n_pos)` vs endogenous `precision × K` |
| `experiment_families.py` / `meta_families.py` | robustness check with random forest + logistic regression |
| `openml_candidates.csv` | the de-duplicated list of OpenML datasets used |
| `results_real3.json` | per (dataset, K) results |
| `paper/` | LaTeX source of the preprint |

## Method in one paragraph

For each dataset we split 50/25/25 (train / validation / test), train a bank of 40 LightGBM
configurations with random hyperparameters (all on log-loss), and sweep the budget *K* to hit target
ratios *n_pos/K* ∈ {0.25 … 16}. We then bootstrap the validation set; for each candidate selection
metric (**precision@k, average precision, ROC-AUC, log-loss**) we pick the argmax configuration and
record its **precision@k on the held-out test set**. Selection quality is summarised as *normalized
regret* (0 = the oracle-best model in the bank, 1 = an average model), pooled across datasets.

## Citation

If you use this, please cite the preprint (see `paper/`). BibTeX will be added on arXiv release.

## License

MIT — see `LICENSE`.
