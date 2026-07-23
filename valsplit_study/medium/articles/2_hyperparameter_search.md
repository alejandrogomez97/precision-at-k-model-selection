# Does fancy hyperparameter search beat a humble grid? A bake-off on 89 datasets

*Part two of a series putting common ML habits to the test on 89 imbalanced datasets. The
habit here: reaching for Optuna "because Bayesian search must beat a grid." I put grid,
random, TPE and CMA-ES — and the multi-fidelity crowd (Hyperband, BOHB) plus a Gaussian-
Process Bayesian sampler — head to head, and measured not just accuracy but time.*

---

## The starting point (the same across the whole series)

The shared rig, briefly:

- **89 imbalanced binary datasets**, metric **Average Precision** (PR-AUC).
- **A fixed, shared test set** touched once. Base learner: **LightGBM** with early stopping;
  per-fold preprocessing, no leakage.
- Search runs in the **E2 framework** — the one Part 1 endorsed: cross-validation over the
  whole development pool, and each configuration is ranked by its **out-of-fold (OOF) AP**
  (no separate validation set; the OOF picks both the number of trees and the winning config).
  The winning config is then refit on all of dev for its final test score. (Part 1 showed you
  can skip that refit under CV and simply *bag* the fold models for a hair more AP; but since
  every method here gets the identical final step, that choice shifts them all equally and
  leaves the ranking untouched — so we hold it fixed.) On 8 representative datasets at 30% and
  100% of dev, 2 seeds.
- **The rule that governs this article: judge at *equal compute*, not equal config count** —
  because, as we'll see, "same number of configs" is emphatically *not* "same time."
- **Every p-value is a paired t-test on the same splits.** In plain terms: I compare two
  strategies *case by case* on the exact same train/test partitions, then ask whether one
  systematically beats the other. Pairing on identical data cancels out the "some datasets are
  just easier" noise, so a small p-value means the winner won on merit — not because it drew a
  luckier split. (Rule of thumb: p = 0.01 means a gap this consistent would happen by chance
  about 1 time in 100.)

---

## The contenders (and the one thing that separates them)

The key difference between search methods is whether the method **learns from its own
results** as it goes:

- **Grid search** — the baseline. Hand-pick a list of configurations up front and try them
  all. Never learns; every config is decided before you see a single result. Systematic,
  dead simple, embarrassingly parallel.
- **Random search** — same "no learning", but samples configurations at random from the
  ranges you give. Sounds naïve, yet it's a famously strong baseline: it spreads its budget
  across the space instead of over a rigid lattice.
- **TPE (Optuna's Bayesian search)** — the first that *learns*: it models which regions tend
  to give good vs bad scores and keeps sampling where things look promising.
- **CMA-ES (an evolution strategy)** — also learns, differently: it keeps a Gaussian "cloud"
  of configs and shifts and shrinks it toward the best performers, generation by generation.

Then the multi-fidelity crowd, popular in Kaggle and AutoML circles, which learns *and*
saves time by killing bad configs early on a cheap "fidelity" (here, few boosting rounds):

- **Hyperband** — races many configs on tiny budgets, keeps the best half, doubles their
  budget, repeats. Random sampling + aggressive early-stopping.
- **BOHB** — Hyperband's early-stopping schedule, but the configs are proposed by Bayesian
  (TPE) sampling instead of at random. "Best of both worlds" on paper.
- **GP Bayesian** — Bayesian optimization with a Gaussian-Process surrogate, the method the
  tree-boosting literature often reports as the strongest full-budget searcher.

## Finding 1 — nobody reliably beats the grid (and the blind ones lose)

Across budgets the seven methods sit within ~0.01 AP of each other, but there's a subtle
pattern worth seeing. At the **smaller budgets (20–80 configs) BOHB posts the best AP** —
smart sampling plus early pruning finds good configs with fewer tries, so it shines when you
can only afford a handful. But its edge is tiny (≤0.006) and never significant. Give everyone
the **full budget (160 configs)** and the plain grid pulls level and nudges ahead (0.707, the
top score) — the fancy searchers have just caught up to where the grid already was.

And at that full budget the only *significant* gaps are the methods that **trail** the grid:
plain random search −0.0104 (p = 0.02) and GP-Bayesian −0.0080 (p = 0.04); TPE is borderline
(−0.0048, p = 0.07); CMA-ES, Hyperband and BOHB tie it. So the honest verdict: **no search
method reliably beats a sensible grid on accuracy** — the clever ones merely reach the same
place, and the blind ones fall behind. And **more budget barely helps** — 20→160 configs
moves AP by a few thousandths at most.

*[TABLE 1 — table2_hpo.png]*

## Finding 2 — "same number of configs" is not "same compute"

Time doesn't rescue the search methods either. On the same 160 configs, the full-budget
methods land in a modest band (TPE 182s, GP 196s, grid 199s, CMA-ES 218s), with **random
search the slowest (260s)** — fitting, since it's also the worst on AP: the true worst of
both worlds. The grid fixes a low `learning_rate` (many boosting rounds), while the learning
samplers drift toward faster-converging configs — but the spread is small, and none of it
buys better AP.

## Finding 3 — the multi-fidelity "speedup" is a mirage under honest CV

This is the twist, and it ties the whole series together. Hyperband and BOHB are supposed to
be the *time* win: train each config on a few trees, **kill the losers early**, only grow the
survivors. And in an **E1 setup** — a single held-out validation set — they are: pruning on
that one `val` signal is cheap, and you'd clock them at an order of magnitude faster.

But Part 1 concluded you should select by **cross-validation (OOF), not a separate val** — so
that's what we do here. And under honest CV, the cheap trick disappears: to prune on the
out-of-fold signal you have to train *all K folds*, so Hyperband and BOHB end up
**2–2.5× *slower* than the grid** (368s and 475s vs 199s) — for no gain in AP. The famous
multi-fidelity speedup was an artifact of the very habit the series argues against; hold
yourself to consistent CV selection and it evaporates.

*[FIGURE 1 — fig_hpo.png]*

*[TABLE 2 — table2b_hpo_time.png]*

## The moral

**Under honest cross-validated selection, a small sensible LightGBM grid is the best deal on
both axes at once** — the highest AP *and* among the fastest to run. Bayesian, evolutionary
and multi-fidelity search don't beat it on accuracy (and random and GP are *significantly
worse*), while multi-fidelity's celebrated time savings vanish the moment you select by CV
instead of a held-out set.

- **Grid: best AP, cheap.** Hard to beat, easy to run.
- **Random / GP: significantly worse AP.** Blind or Bayesian, they lose here.
- **Hyperband / BOHB: no AP gain and 2–2.5× slower** once pruning has to run on OOF.

So don't reach for fancy hyperparameter search to squeeze more out of a well-chosen grid — on
this problem it doesn't help and can hurt. Save it for spaces a grid genuinely can't cover.

*Next in the series: the ensemble myth — is "a thousand LightGBM configs" really faster than
building an ensemble? Code and all the numbers are in the study repo; every p-value is a
paired t-test on the same splits.*
