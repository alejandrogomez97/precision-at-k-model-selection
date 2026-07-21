# One validation set, or two? Five machine-learning habits, put to the test on 89 datasets

*A few years ago I wrote a short Medium post with a slightly heretical claim: you don't
need a separate held-out set to pick the number of trees and another one to pick the
model — cross-validation can do both at once. It was mostly a theoretical argument.
This time I wanted proof. And once I had the machinery running, I couldn't resist
throwing four more common ML habits into the fire.*

---

## Why bother?

Most of us carry a bag of habits we never actually tested. We split off a validation
set "because that's what you do". We retrain the final model on all the data "to be
safe". We reach for Optuna "because Bayesian search must beat a grid". We tell ourselves
"I'll just try a thousand LightGBM configs, it's faster than building an ensemble". And
some of us quietly believe that adding a few invented features never hurts — "the model
will just ignore the useless ones, and maybe one turns out to be gold".

Are any of these true? I ran the numbers on **89 imbalanced binary datasets** (from
imblearn and OpenML), and the answers are more interesting — and more humbling — than I
expected.

> **The one methodological rule that runs through everything below:** compare *at equal
> time*, not at an equal number of configurations. Half of the folk wisdom in ML falls
> apart the moment you hold wall-clock time constant instead of "number of models tried".

### The setup (once, for all experiments)

- **89 imbalanced binary datasets.** Metric: **Average Precision** (PR-AUC), the right
  choice under class imbalance.
- **A fixed, shared test set** (30%, capped at 6,000 rows) that every strategy is judged
  on — touched once.
- **A fraction sweep instead of absolute sizes.** The whole point is to see how each
  strategy behaves **when you have lots of training data versus very little** — that's
  often what decides which approach is best. So instead of fixed "300, 600, 1200… rows", I
  vary the *fraction* of each dataset's development pool (10%, 20%, …, 100%) and watch
  performance grow with data. Why fractions and not absolute counts? Because datasets have
  different sizes: sweeping by absolute count means only the big datasets reach the large
  sizes — and if those big datasets happen to be the hard ones, your aggregate curve dips
  at the end and lies to you. With fractions, **every dataset is present at every point**,
  so the curve reflects data quantity, not which datasets survived. (I learned this the
  hard way — my first plots showed performance *dropping* with more data. Pure composition
  artifact.)
- Base learner: **LightGBM**, 18 grid candidates (6 configs × 3 imbalance techniques),
  number of trees always set by early stopping. Per-fold preprocessing, no leakage.

*[FIGURE 1 — fig_A_story.png]*

---

## Chapter 1 — The original heresy: one validation set, or two?

Here are the two contenders, both starting from the same development pool and judged on
the same test set:

- **E1 — separate validation.** Split dev into `train` + `val`. Cross-validate *within
  train* to fix the number of trees (early stopping). Then pick the best candidate on the
  independent `val` set. This is the "two sets" school.
- **E2 — CV only.** Run cross-validation over all of dev; the out-of-fold predictions
  decide **both** the number of trees **and** the winning candidate. This is the "let CV
  do everything" school — the one my old post defended.

Across **1,780 paired comparisons**, the verdict is sober but clear: **the two are almost
identical, with a small, systematic edge to E2** — mean ΔAP(E1−E2) = **−0.0037**,
**p = 9×10⁻⁷**. And the edge has a shape: with **scarce data** E2 wins clearly (≤30% of
the pool: −0.0049, **p = 0.006**); with **abundant data** it narrows to a tie (≥80%:
−0.0018, p = 0.09). The gap also grows with class imbalance — exactly where E1's tiny
`val` set has too few positives to choose well.

**The takeaway:** carving off a separate validation set buys you nothing, and quietly
costs you when data is scarce or imbalanced. My years-old claim survives contact with the
data. Use CV.

**How it was done.** E1: split dev into `train` (75%) + `val` (25%); 4-fold CV on `train` fixes the number of trees (early stopping on Average Precision, up to 3,000 trees, patience 50); the 18 candidates (6 LightGBM configs — num_leaves ∈ {15,31,63} × reg_lambda ∈ {0,1} — × 3 imbalance techniques: none / class_weight / SMOTE) are scored on `val`; the winner is refit on `train`+`val`. E2: 4-fold CV over all of dev; the out-of-fold predictions pick both trees and candidate; the winner is refit on dev. 89 datasets × 10 fractions × 2 seeds = 1,780 paired cells; per-fold preprocessing (impute + scale + one-hot), no leakage; every p-value is a paired t-test.

*[TABLE 1 — table1_e1_vs_e2.png]*

---

## Chapter 2 — "I always retrain on everything." Should you?

A near-universal habit: once you've selected your model, refit it on *all* the data
before shipping. Free lunch, right?

I evaluated four variants — E1/E2 × with/without a final retrain — on the same splits.
The answer **depends on the strategy**, and it's a lovely little illustration of *why*
you did the selection:

- **E1: yes, retrain.** ΔAP = **+0.0091**, **p = 2×10⁻³⁶**. E1 held out a `val` set it
  never trained on — retraining on `train`+`val` hands that data back. E1-without-retrain
  is the worst of the four.
- **E2: don't bother — it can even hurt.** ΔAP = **−0.0036**, **p = 1×10⁻¹⁰**. CV already
  used every example; the honest "no-retrain" here is a *bagged* ensemble of the fold
  models, and it slightly beats the refit.

**The takeaway:** retraining isn't a virtue in itself — it's a way to recover data you
held out. If you didn't hold any out (CV), there's nothing to recover.

**How it was done.** Same 18-candidate setup as Chapter 1, evaluating all four combinations (E1/E2 × retrain/no-retrain) on identical splits. "No-retrain" means: for E1, keep the model trained on `train` only; for E2, a *bagged* ensemble of the 4 fold models (no refit on dev). 89 datasets, paired t-test.

*[FIGURE 2 — fig_retrain.png]*

---

## Chapter 3 — Optuna vs a humble grid (and does more searching help?)

Surely Bayesian optimization beats a hand-picked grid? I compared, **at an equal number
of configurations (18 vs 18)**, a fixed grid against Optuna's TPE. Then I widened it:
**grid vs random search vs TPE vs CMA-ES**, at budgets of 20, 40, 80 and 160
configurations.

A quick word on the contenders — the key difference is whether the method *learns from
its own results* as it goes:

- **Grid search** — the baseline. You hand-pick a list of configurations up front and try
  them all. It never learns: every config is decided before you see a single result.
  Systematic, dead simple, embarrassingly parallel.
- **Random search** — same "no learning", but instead of a fixed list it samples
  configurations at random from the ranges you give. Sounds naïve, yet it's a famously
  strong baseline: it spends its budget spread across the space instead of on a rigid lattice.
- **TPE (Optuna's Bayesian search)** — the first one that *learns*. It builds a running
  model of which regions of the space tend to give good vs bad scores, and keeps sampling
  where things look promising. Exploits what it has seen so far.
- **CMA-ES (an evolution strategy)** — also learns, but differently: it keeps a Gaussian
  "cloud" of configurations and, generation after generation, shifts and shrinks that cloud
  toward the best performers. Strong on smooth, continuous spaces.

The interesting question isn't "which is fanciest" but "does *learning as you go* actually
beat blindly trying points?" Two findings, both anticlimactic in the best way:

1. **At equal budget, nobody pulls away from the grid.** On a balanced panel, all four
   land within ~0.005 AP of each other, and none of the search methods beats the grid
   *significantly* (random +0.0025, p=0.40; TPE +0.0030, p=0.32; CMA-ES +0.0036, p=0.21).
   They win a few more times in win-rate, but the effect is a rounding error.
2. **More budget barely moves the needle** (max gain from 20→160 configs: ~+0.006 AP).

A methodological aside worth its own paragraph: *"same number of configurations" is not
"same compute."* The grid fixes `learning_rate = 0.05`, so every model grows many boosting
rounds before early stopping. Optuna samples higher learning rates that converge in fewer
trees — so Optuna-18 is actually *faster* than grid-18. Which is exactly why the honest
way to compare is by time, not by count. Hold that thought.

**The takeaway:** for a well-chosen small LightGBM grid, fancier search — Bayesian or
evolutionary — isn't worth its cost. Random search is a shockingly good baseline.

**How it was done.** 8 representative datasets, fractions 30% & 100%, 2 seeds, in the E1 framework. The three Optuna samplers (random, TPE, CMA-ES) search a full **8-hyperparameter space** (num_leaves 7–255, min_child_samples 5–100, reg_lambda & reg_alpha 1e-3–10, colsample_bytree 0.5–1, subsample 0.6–1, learning_rate 0.01–0.2, and the imbalance technique). The grid is a fixed, shuffled **360-config grid** covering **5 of those knobs** (num_leaves, min_child_samples, learning_rate, reg_lambda and the imbalance technique — the other three left at LightGBM defaults). Budgets of 20/40/80/160 configs are read off a single 160-config run (best-so-far checkpoints). Compared on a balanced panel of the 32 cells common to all four methods; paired t-test vs the grid. The verdict is unambiguous: the *best* of the search methods, **CMA-ES, beats the grid by a mere +0.0036 AP — and even that is not significant (p = 0.21), winning just 53% of cells**. Random search (+0.0025, p = 0.40) and TPE (+0.0030, p = 0.32) fare no better.

*[FIGURE 3 — fig_hpo.png]*

*[TABLE 2 — table2_hpo.png]*

And what about *time*? Since AP doesn't separate them, maybe wall-clock does. It does — a
little, and in a revealing direction. The ranking is stable at every budget: **random
search is the slowest, CMA-ES the fastest**, with grid and TPE in between — a **~25–30%
spread** in wall-clock *for the exact same number of configs*. Why? The Chapter-3 aside
again: random samples learning rates uniformly and keeps drawing slow-to-train configs
(low learning rate → many boosting rounds), while CMA-ES drifts toward the cheap,
high-learning-rate regions. The punchline for the time-conscious: random search isn't just
no better on AP — it's the **most expensive per config**, and comparing "at equal number of
configs" quietly gifts it ~25% more wall-clock than CMA-ES… and it *still* doesn't win.
None of it rescues the search methods: on a small LightGBM space the grid remains the best
AP-per-second deal.

*[TABLE 2b — table2b_hpo_time.png]*

---

## Chapter 4 — The ensemble myth: "LightGBM is faster than an ensemble"

This is the one I hear most, and on its face it's *true*: a **single** LightGBM trains far
faster than a 10-model blend. So people skip ensembles on principle — "too slow". But here's
the catch: **nobody trains just one LightGBM.** They try dozens, hundreds, a thousand
configs, sweeping hyperparameters — and *that* is where the time goes. Once you add up all
those configs, you've spent as much compute as an ensemble would have cost… you just have
nothing better than a single tuned LightGBM to show for it. So the honest question isn't
"single model vs ensemble" — it's: **given the time you were going to burn on all those
configs anyway, what gives the best AP?**

First, the ensemble. I took a pool of **10 diverse model families** (logistic regression,
Gaussian NB, kNN, LightGBM, HistGBM, XGBoost, ExtraTrees, Random Forest, CatBoost, MLP —
one fixed config each, *no* per-family tuning) and combined them with a **greedy Caruana
blend** — a weighted average of probabilities whose weights are chosen by hill-climbing.
Two symmetric flavors, mirroring E1 and E2:

- **ens-E1:** learn the blend on the separate `val`. Trains each family once → cheap.
- **ens-E2:** learn the blend on the CV out-of-fold predictions and refit the families on
  all dev. That's 4 folds × 10 families + 10 refits = **50 trainings** vs 10 → several times slower.

The headline: **ens-E2 beats plain E2 in ~2 of 3 cases (ΔAP +0.012, p = 2×10⁻⁸)** — a real,
significant improvement — but at several times the time. ens-E1, at equal time, doesn't
beat E1.

*[FIGURE 4 — fig_C_isotime.png]*

So ensembles *do* help — but they're slower, and the "many configs" crowd would say that
settles it. It doesn't. **Let's give the cheap strategies ens-E2's entire time budget**
(call it t*) and let them do *more*: more grid configurations for E1 and E2 (E1@t*, E2@t*),
more members for the ensemble (ens-E1@t*). Same time, best AP wins.

**The ensemble still wins.** And here I made the comparison genuinely fair — each @t*
strategy now spends *the same seconds* ens-E2 does (at 100% of dev: E1@t* 352s, E2@t* 358s,
ens-E2 355s). Even so, ens-E2 beats grown ens-E1, E1@t* and E2@t* at **every data
fraction**, in **75–87% of cells** (ens-E2 − ens-E1@t*: +0.018, p=2×10⁻¹⁶; − E1@t*:
+0.013, p=1×10⁻⁶; − E2@t*: +0.010, p=2×10⁻⁵). Giving the cheap strategies more time to try
more configs lets them win a little more often — but they don't catch up. ens-E2's edge is
*structural, not a matter of time*: it uses all of dev both to learn the blend (OOF) and
for the final models (refit). Throwing more configs at a single LightGBM saturates.

*[FIGURE 5 — fig_isofinal.png]*

*[TABLE 3 — table3_isotime.png]*

The raw timings tell the cost side of the story — and hold a twist. **ens-E1 is actually
the cheapest strategy of all** (it trains each family just once: ~19s on average, less than
plain E1 or E2), while **ens-E2 is the priciest, roughly 5× the base strategies** (it pays
for 4 CV folds × 10 families plus 10 refits). That gap is exactly why the equal-time
comparison matters. To make it scrupulously fair, each cheap strategy is grown until it
**crosses** ens-E2's time budget (t*): E1@t* and E2@t* keep adding grid configs, and
ens-E1@t* keeps adding members until it too has spent the full t*. So all three genuinely
get ens-E2's wall-clock. And still — even lavished with the same time and dozens of extra
members — **ens-E1@t* loses to ens-E2 at every fraction**. The extra time simply can't buy
the cheap ensemble the structural advantage ens-E2 gets from using all of dev twice over:
once (via OOF) to learn the blend, and again to refit the final members.

*[TABLE 3b — table3b_ens_time.png]*

**The takeaway:** the "many LightGBM configs is faster" argument is a false economy. Those
configs cost the same time as an ensemble — and at equal time, the ensemble wins. (One
practical nuance: for the ensemble, the retrain-vs-bagged distinction that mattered for a
single model washes out — a 10-model blend already tames variance — so use the cheaper,
bagged version.)

**How it was done.** 16 datasets, 10 fractions, 2 seeds. Pool of 10 families, **one fixed config each, no tuning**: logistic regression (balanced), GaussianNB, kNN(k=25), LightGBM(400 trees, num_leaves 31), HistGBM(400), XGBoost(400, depth 6), ExtraTrees(300), RandomForest(300), CatBoost(400, depth 6), MLP(64-32). The blend is greedy Caruana selection with replacement, maximizing AP (it usually keeps 2–4 families). **Equal-time:** t* = the wall-clock ens-E2 spends per cell; E1@t* and E2@t* keep adding LightGBM grid configs (up to 800) and ens-E1@t* keeps adding pool members until they've spent t* (taking the config that *crosses* t*, so they get the full budget); 8 datasets, paired t-test.

---

## Chapter 5 — Speculative feature engineering: does the noise hurt?

Last one, and my favorite because the belief is so widespread: *"tree models discard
irrelevant features, so inventing variables never hurts — and once in a while you stumble
on a useful one."*

I put it to the test with the strongest tree strategy (E2 LightGBM). At 100% of dev, I
added **random, meaningless features** — arithmetic mashups of random feature pairs
(`v_i/v_j`, `v_i·v_j`, `v_i−v_j`, …) and pure noise — in growing amounts: 10%, 20%, …, up
to 100% (at 100%, half the columns are invented garbage). Then I watched the AP.

The result is a nuanced "mostly true": **the tree does largely ignore the noise.** AP stays
almost flat as you pile it on — until, at 100% noise, there's a **small, borderline drag**
(ΔAP ≈ −0.004, p = 0.066), driven mostly by a couple of sensitive datasets while the rest
shrug it off. And crucially, **there was no free lunch**: on average the random features never
*improve* things by luck. Training time, of course, rises (more columns to chew through)
even though quality doesn't.

**The takeaway:** the belief is *mostly* right — LightGBM won't collapse under nonsense
features — but "it never hurts" is too strong. The real damage isn't the AP: even though
the drag at 100% noise isn't statistically significant, **the model takes noticeably longer
to train for exactly zero gain in AP**. And that's the whole problem — under the equal-time
lens from Chapter 4, that wasted time is time you could have spent training *more* models
(a bigger grid, more ensemble members), which we already know *does* help. So speculative
feature engineering is a strictly worse use of your compute: you pay in time and get nothing
back. Spend that effort — and that time — on features that mean something.

**How it was done.** The question is about *tree* models, so the model is E2 with LightGBM (ens-E2 is avoided: its non-tree members — logistic, kNN, MLP — do suffer from noise and would confound the answer). 8 datasets, 100% of dev (capped at 8,000 rows for cost), 2 seeds. The invented features are random meaningless combinations of feature pairs — `v_i/v_j`, `v_i·v_j`, `v_i−v_j`, `|v_i−v_j|`, `v_i+v_j` — plus pure noise, added in growing amounts (10%…100% of the original feature count; at 100% half the columns are invented). We report the paired ΔAP vs 0% noise.

*[FIGURE 6 — fig_feateng.png]*

*[TABLE 4 — table4_feateng.png]*

---

## The moral of the story

Five habits, five verdicts:

1. **One validation set is enough** for imbalanced data (and marginally better than two).
   CV can pick the trees and the model. *p = 9×10⁻⁷.*
2. **Retraining** is only worth it if you held data out. With CV you didn't — so skip it.
   *E1 p = 2×10⁻³⁶; E2 p = 1×10⁻¹⁰.*
3. **A small, sensible grid** matches Bayesian and evolutionary search on AP; more budget
   barely helps (*all method gaps p > 0.2*). The one thing that *does* differ is **time**:
   the *learning* optimizers (TPE, CMA-ES) steer toward fast-training configs and away from
   the slow, low-learning-rate ones that never pay off, so they're the cheapest per config —
   while **random search is the worst of both worlds**, no better on AP and the slowest to run.
   If you're going to search at all, use a learning optimizer, not random.
4. **At equal time, ensembles beat "many LightGBM configs."** The configs cost the same
   time and lose. *ens-E2 wins 75–87%, p ≤ 2×10⁻⁵.*
5. **Speculative feature engineering** mostly doesn't hurt (trees ignore noise) but doesn't
   help either — and it still *costs you*: more columns mean longer training for zero AP
   gain, and under the equal-time lens that time would have bought you more models. A
   strictly worse use of compute. *ΔAP ≈ −0.004 at 100% noise, p = 0.07.*

If there's a single thread, it's this: **measure at equal time, and be suspicious of the
habit you never tested.** Most of what we do out of reflex is neither as helpful nor as
harmful as we assume — and the few things that genuinely move the needle (an honest
ensemble, real feature engineering) are the ones worth your compute.

*Methodology, code and all the numbers behind every claim are in the study repo. Every
p-value is a paired t-test on the same splits.*
