# One validation set, or two? An empirical test on imbalanced data

*A follow-up to my earlier, mostly theoretical argument that a single validation
set is enough for both early stopping and hyperparameter selection — now stress-tested
on 93 imbalanced binary datasets.*

---

## The question

In a previous article I argued, mostly on theoretical grounds, that **a single
validation set is enough** to do both *early stopping* (choosing the number of
trees/iterations) and hyperparameter selection: early stopping is just preselection
over one more hyperparameter, so keeping two separate sets is a response to data
scarcity, not a methodological necessity.

Here I test that **empirically**, and I measure what happens **as a function of how
much data you have**. The intuition to falsify is twofold and seemingly contradictory:

- With **lots of data**, an independent validation set for *selection* should reduce
  the **winner's curse** (the optimistic bias of picking the best among many
  candidates scored on the same data) and generalize better.
- With **little data**, that held-out set is tiny and noisy; using all the
  information through cross-validation (CV) to decide everything should pay off.

## The two strategies

Both start from the same development pool `dev` and the **same test set** (fixed and
shared at each point), and share the same candidate space. The **only** difference is
the selection mechanism, so the experiment isolates the winner's curse from selection
noise.

**E1 — separate validation.** `dev → train + val`. Each candidate's number of trees
is fixed by **CV inside `train`** (early stopping). The best candidate is chosen on the
**independent `val` set** (not used for early stopping).

**E2 — CV only.** `dev → CV`. The out-of-fold (OOF) predictions decide **both** the
number of trees and the best candidate at once.

On top of each, two ways to build the **final model** that goes to test — which lets us
measure whether retraining is worth it:

- **With retraining:** E1 refits the winner on `train+val`; E2 refits on `dev`. Uses
  all the data, but throws away the models already trained.
- **Without retraining:** E1 uses the model trained on `train` only; E2 uses a
  **bagged ensemble** of the *k* fold models. Reuses what's trained, with less data per
  model.

With retraining, both strategies use the **same amount of data** for the final model —
only the selection signal differs. Without retraining, each uses less data but skips the
refit.

## Setup

- **Data:** imbalanced binary datasets (imblearn + OpenML), varied prevalences and
  imbalance ratios. LightGBM as the base learner.
- **Fixed test:** 30% of each dataset (capped at 6,000 rows), identical for E1 and E2.
  Only the development pool shrinks.
- **Data sweep:** `dev ∈ {300, 600, 1200, 2500, 5000, 10000, 20000}`, stratified
  subsampling, multiple seeds.
- **Candidates:** 6 LightGBM configs × 3 imbalance techniques (none / class_weight /
  SMOTE) = 18. Early stopping on `average_precision`, patience 50, up to 3,000 trees.
- **Primary metric:** Average Precision (PR-AUC), the right choice under imbalance;
  log-loss also recorded. Preprocessing fit per-fold (no leakage). All timings logged.

---

## Result 1 — E1 vs E2: it barely matters

With **93 datasets, 3 seeds and 1,127 paired comparisons**, the result is as clear as
it is sober: **the two strategies are practically equivalent.**

- The median AP difference between E1 and E2 is **0.0004**; in **54%** of cells it is
  below 0.01 AP. For practical purposes, it doesn't matter which you use.
- There is, however, a **small but systematic and statistically detectable edge for E2**
  (single split, CV decides everything): mean ΔAP = **−0.0024** in E2's favor,
  *t* = −2.42, **p = 0.016**; E1 win-rate = 0.446.
- E2's edge **grows with imbalance**: from ~0 at ratio < 5 to **−0.0037** at ratio > 20.
- And it is **largest with little data**: at `dev = 300` the gap is −0.0056 (≈ 2.3 SE).
- **The regime we anticipated — E1 winning with lots of data thanks to a smaller
  winner's curse — does not appear.** E1 never clearly beats E2 at any size.

| dev size | AP E1 | AP E2 | ΔAP (E1−E2) | E1 win-rate | n |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.502 | 0.507 | −0.0056 | 0.39 | 265 |
| 600 | 0.540 | 0.541 | −0.0002 | 0.44 | 268 |
| 1200 | 0.596 | 0.597 | −0.0008 | 0.54 | 226 |
| 2500 | 0.593 | 0.597 | −0.0038 | 0.43 | 163 |
| 5000 | 0.553 | 0.555 | −0.0018 | 0.46 | 96 |
| 10000 | 0.536 | 0.539 | −0.0025 | 0.39 | 62 |
| 20000 | 0.453 | 0.454 | −0.0006 | 0.47 | 47 |

*(Absolute AP drops past 5,000 not because models get worse, but because only the large
— and harder — datasets reach those sizes: a sample-composition effect. The valid
comparison is the **paired** one, within each dataset.)*

> **[FIGURE 1 — fig_A_story.png]** *Three panels: near-identical performance (left),
> E2's edge vs data size (center), E2's edge vs imbalance (right).*

**Reading.** Holding out a separate validation set (E1) buys nothing here: since the
final model is retrained on all of `dev` in both strategies, E1 only contributes an
*unbiased but smaller and noisier* selection signal, while E2 uses all the data through
OOF. The winner's-curse penalty of reusing the folds for both trees and model turns out
**negligible** with 18 candidates, and it doesn't offset the cost of carving up the
data. The scarcer or more imbalanced the problem (fewer positives in E1's `val`), the
more it shows.

## Result 2 — Does retraining the final model help? It depends on the strategy

Evaluating all four variants (E1/E2 × with/without retraining) on the same splits
isolates the value of the final refit.

- **E1: yes, clearly.** ΔAP(retrain − noretrain) = **+0.0173**, retrain wins 64% of the
  time, *t* = 5.34, **p = 3×10⁻⁷**. E1 *must* retrain — otherwise it wastes the `val`
  data, and **E1-without-retraining is the worst of the four variants.**
- **E2: indifferent.** ΔAP = **−0.0032**, *p* = 0.14 (not significant), a whisker in
  favor of *not* retraining. The bagged CV ensemble already captures what a refit would
  give — and you've already trained those models.

> **[FIGURE 2 — fig_retrain.png]** *The four variants (left); the effect of retraining
> per strategy (right). E1-noretrain sits clearly lowest; E2-bagged rides with the best.*

Practical takeaway: **if you use a separate val set, retrain on train+val. If you use
CV, don't bother refitting — keep the bagged fold models.**

## Result 3 — Optuna vs grid: not worth it here

Replacing the 18-candidate grid with **Optuna (TPE, 40 trials)** over a wider LightGBM
space, in both strategies:

- On average Optuna **does not beat** the grid. At small sizes (300–600) it is often
  **worse** (more search variance), and only marginally better at some larger sizes.
- It costs **~10× more time** (e.g. at `dev = 20000`: ~570 s vs ~75 s of search).

| dev | AP grid | AP optuna | Δ | search grid | search optuna |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.525 | 0.514 | −0.012 | 1.8 s | 24 s |
| 1200 | 0.624 | 0.628 | +0.005 | 8 s | 57 s |
| 5000 | 0.652 | 0.652 | −0.001 | 25 s | 176 s |
| 20000 | 0.499 | 0.501 | +0.002 | 80 s | 570 s |

> **[FIGURE 3 — fig_B_optuna.png]** *Test AP, gain, and search-time cost, grid vs Optuna.*

For a well-chosen small grid on LightGBM, Bayesian search is a poor trade.

## Result 4 — Many families + ensembles: better, but not faster

Instead of many LightGBMs, a **pool of families** (logistic regression, RF, ExtraTrees,
HistGBM, LightGBM, XGBoost, CatBoost, MLP, kNN, GNB) plus a **greedy ensemble
(Caruana selection)** on `val`. We track the *anytime* curve — best test AP reached as a
function of cumulative time — against the E1-grid reference.

- The greedy ensemble improves over the **best single family** in its own pool, but it
  **does not beat** the focused E1-grid LightGBM, and its edge shrinks as data grows
  (ens ≥ E1 win-rate: 0.50 at 1,200 → 0.25 at 20,000).
- It costs **10–20× more time**; the fraction of cases where it beats E1 *in equal or
  less time* falls from 0.43 to 0.00.

> **[FIGURE 4 — fig_C_families.png]** *Test AP, win-rate vs E1-grid, and val→test
> consolidation of the ensemble.*

Throwing families and blends at the problem is not a shortcut to *better and faster*
here — a single well-tuned gradient-boosting model with a small grid is more efficient.

---

## Conclusions

1. **A single validation set is enough** — and, if anything, marginally better — for
   imbalanced data. The theoretical claim holds up empirically.
2. The difference is **practically irrelevant** (median 0.0004 AP); choose by
   convenience or data budget.
3. If you are **data-scarce or highly imbalanced**, lean toward **E2** (let CV decide
   everything): it uses every positive.
4. **Retraining** the final model helps E1 (recovers the val data) but not E2 (bagging
   already covers it). E1-without-retraining is the worst option.
5. On this problem, **neither Optuna nor a families+ensemble pipeline pays for its extra
   cost** over a small, well-chosen LightGBM grid.

*Methodology, code and data: `precision-at-k-study/valsplit_study`. Results 1–4 from
93 / 93 / 16 / 16 datasets respectively; the retraining comparison from the seed-0 pass
over 93 datasets.*
