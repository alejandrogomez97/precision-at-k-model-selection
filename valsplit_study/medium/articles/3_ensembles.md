# "LightGBM is faster than an ensemble." True — and exactly why the ensemble wins

*Part three of a series putting common ML habits to the test on 89 imbalanced datasets.
This is the habit I hear most, and it's the most seductive because its premise is actually
true: a single LightGBM really is faster than a 10-model blend. The mistake is what people
do next.*

---

## The starting point (the same across the whole series)

The shared rig, briefly:

- **89 imbalanced binary datasets**, metric **Average Precision** (PR-AUC).
- **A fixed, shared test set** touched once; a **fraction sweep** (10%…100% of each
  dataset's development pool) so we see behavior with little and lots of data.
- Base learner: **LightGBM** with early stopping; per-fold preprocessing, no leakage.
- **The rule that governs this whole article: compare at *equal wall-clock time*, not at an
  equal number of models.** Everything below hinges on it.
- **Every p-value is a paired t-test on the same splits.** In plain terms: I compare two
  strategies *case by case* on the exact same train/test partitions, then ask whether one
  systematically beats the other. Pairing on identical data cancels out the "some datasets are
  just easier" noise, so a small p-value means the winner won on merit — not because it drew a
  luckier split. (Rule of thumb: p = 0.01 means a gap this consistent would happen by chance
  about 1 time in 100.)

---

## The myth, stated fairly

"I don't build ensembles — I just train a LightGBM, it's way faster." On its face, **true**:
one LightGBM trains far faster than a blend of ten models. So people skip ensembles on
principle.

But here's the catch: **nobody trains just one LightGBM.** They try dozens, hundreds, a
thousand configs, sweeping hyperparameters — and *that* is where the time goes. Once you add
up all those configs, you've spent as much compute as an ensemble would have cost… you just
have nothing better than a single tuned LightGBM to show for it.

So the honest question isn't "single model vs ensemble". It's: **given the time you were
going to burn on all those configs anyway, what gives the best AP?**

## The ensemble

I took a pool of **10 diverse model families** (logistic regression, Gaussian NB, kNN,
LightGBM, HistGBM, XGBoost, ExtraTrees, Random Forest, CatBoost, MLP — one fixed config
each, *no* per-family tuning) and combined them with a **greedy Caruana blend**: a weighted
average of probabilities whose weights are chosen by hill-climbing. Two symmetric flavors,
mirroring the E1/E2 split from part one:

- **ens-E1:** learn the blend on a separate `val`. Trains each family once → cheap.
- **ens-E2:** learn the blend on the CV out-of-fold predictions and refit the families on
  all dev. That's 4 folds × 10 families + 10 refits = **50 trainings** vs 10 → several times
  slower.

The headline: **ens-E2 beats plain E2 in about two of every three cases** (ΔAP **+0.012**,
p = **2×10⁻⁸**) — a real, significant improvement — but at several times the time. ens-E1, at
equal time, doesn't beat E1.

*[FIGURE 1 — fig_C_isotime.png]*

## The fair fight: give everyone the ensemble's time budget

Ensembles help but cost more, and the "many configs" crowd would say that settles it. It
doesn't. So I ran the scrupulously fair experiment: take ens-E2's wall-clock time per
problem (call it **t\***) and give it to every cheaper strategy, letting each **grow until
it crosses t\***:

- **E1@t\*** and **E2@t\*** keep adding LightGBM grid configs until they've spent t\*.
- **ens-E1@t\*** keeps adding ensemble members (from an extended pool) until it too has spent
  t\*.

So all of them get *exactly* ens-E2's wall-clock — a genuinely equal-time comparison, no
strategy short-changed.

**ens-E2 still wins — at every data fraction, and by a significant margin.** Handed the same
seconds and dozens of extra configs or members, none of the cheaper strategies catches it:
ens-E2 beats grown **ens-E1@t\*** by +0.0147 AP (winning **87%** of cells, p = 4×10⁻¹⁷),
**E1@t\*** by +0.013 (**75%**, p = 1×10⁻⁶) and **E2@t\*** by +0.010 (**76%**, p = 2×10⁻⁵).
The advantage isn't about time; it's **structural**: ens-E2 uses all of dev *twice over* —
once via out-of-fold predictions to learn the blend, and again to refit the final members.
Throwing more configs at a single LightGBM, or more members at a blend trained on a held-out
slice, simply saturates.

*[FIGURE 2 — fig_isofinal.png]*

*[TABLE 1 — table3_isotime.png]*

## Where the time actually goes

It's worth seeing the cost side directly, because it holds a twist:

- **ens-E1 is the *cheapest* strategy of all** — it trains each family just once, so it's
  even cheaper than plain E1 or E2.
- **ens-E2 is the priciest, ~5× the base strategies** — the price of using all of dev for
  both the blend and the refit.
- When over-funded to t\*, E1@t\* and E2@t\* spend it on more grid configs, and ens-E1@t\*
  spends it on more members — and it *still* loses. Cheap-and-plentiful can't buy the
  structural edge.

*[TABLE 2 — table3b_ens_time.png]*

**The takeaway:** the "many LightGBM configs is faster" argument is a false economy. Those
configs cost the same time as an ensemble — and at equal time, the ensemble wins. (One
practical nuance: for the ensemble, the retrain-vs-bagged distinction that mattered for a
single model washes out — a 10-model blend already tames variance — so use the cheaper,
bagged version.)

**How it was done.** 16 datasets (8 for the equal-time comparison), 10 fractions, 2 seeds.
Pool of 10 families, one fixed config each, no tuning. The blend is greedy Caruana selection
with replacement, maximizing AP (it usually keeps 2–4 families). Equal-time: t\* = the
wall-clock ens-E2 spends per cell; E1@t\*, E2@t\* and ens-E1@t\* each keep adding
configs/members until they *cross* t\*, so all three genuinely spend ens-E2's full time.
Paired t-tests.

---

## The moral

**At equal time, an honest ensemble beats "more of a single model."** The seductive part of
the myth — that one LightGBM is faster — is true and irrelevant, because you never stop at
one. Spend the compute you were going to spend anyway on a small, diverse blend evaluated
out-of-fold, and you get a real, significant lift that no amount of extra hyperparameter
search matches.

*Next in the series: does speculative feature engineering hurt? Code and all the numbers are
in the study repo; every p-value is a paired t-test on the same splits.*
