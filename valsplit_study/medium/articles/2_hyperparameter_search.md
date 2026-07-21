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
- Search runs inside the **E1 framework** (dev → train/val; trees fixed by CV on train;
  candidates scored on val), on 8 representative datasets at 30% and 100% of dev, 2 seeds.
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

## Finding 1 — on accuracy, nobody pulls away from the grid

At an equal number of configurations, all **seven** methods land within ~0.005 AP of each
other, and **none beats the grid significantly**: random +0.0025 (p = 0.40), TPE +0.0030
(p = 0.32), CMA-ES +0.0036 (p = 0.21), GP +0.0045 (p = 0.30), BOHB +0.0038 (p = 0.37),
Hyperband +0.0015 (p = 0.58). GP nudges out the highest absolute AP (0.703 vs the grid's
0.698), but even that is a rounding error, not significance — every gap has p > 0.2.

And **more budget barely helps**: going from 20 to 160 configurations moves AP by about
+0.006 at most.

*[TABLE 1 — table2_hpo.png]*

## Finding 2 — "same number of configs" is not "same compute"

If accuracy doesn't separate them, *time* does — and this is the methodological punchline.
The grid fixes a low `learning_rate`, so every model grows many boosting rounds before early
stopping. The smart samplers drift toward higher learning rates that converge in fewer trees,
so they're cheaper per config. The standout among the full-budget methods is **GP: the same
160 configs, but ~2.5× faster than the grid (225s vs 565s)**, because it homes in on
efficient regions. Random search, sampling blindly, is the *slowest* of all (681s) — the
worst of both worlds: no better on AP, most expensive to run.

## Finding 3 — where fancy search finally earns its keep: multi-fidelity

Here's the real time story. Hyperband and BOHB don't run every config to completion — they
train each on just a few trees first and **kill the losers early**, letting only the
promising ones grow. On the *exact same 160 configurations*, the payoff is dramatic:

> **Hyperband and BOHB reach the same AP as everyone else in ~15× less wall-clock** —
> 38s and 37s, versus the grid's 565s.

Look at the time panel (log scale): the two multi-fidelity methods sit an order of magnitude
below the rest. And it's not a quality trade-off — BOHB even matches the full-budget methods'
AP (0.702), because its Bayesian sampling spends the time it saves on *better* configs rather
than just more of them. This is the one place fancier search clearly wins: **not more
accuracy — the same accuracy, far cheaper.**

*[FIGURE 1 — fig_hpo.png]*

*[TABLE 2 — table2b_hpo_time.png]*

## The moral

**For a small, sensible LightGBM grid, no search method buys you more accuracy.** Bayesian,
evolutionary and multi-fidelity all tie the grid on AP (*every gap p > 0.2*), and more budget
barely moves the needle. What genuinely differs is **time** — and there the verdict is clear:

- **Multi-fidelity (Hyperband, BOHB) gets the same AP in ~15× less time** by pruning bad
  configs after a few trees. If search time hurts, this is the win.
- **GP** is a solid ~2.5× faster than the grid at full budget.
- **Random search is the worst of both worlds** — no better on AP, and the slowest to run.

So don't expect fancy search to raise your score over a well-chosen grid — it won't. But if
you care about the clock, skip random and reach for a multi-fidelity method: same score, a
fraction of the compute.

*Next in the series: the ensemble myth — is "a thousand LightGBM configs" really faster than
building an ensemble? Code and all the numbers are in the study repo; every p-value is a
paired t-test on the same splits.*
