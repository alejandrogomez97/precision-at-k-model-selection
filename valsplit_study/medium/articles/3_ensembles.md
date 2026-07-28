# Training a thousand LightGBMs is a waste of time — build the ensemble

*Part three of a series putting common ML habits to the test. LightGBM's one selling point is
speed, so training a thousand of them to hunt for a good config is self-defeating: you burn an
ensemble's worth of compute and still cap out at a single model's ceiling — the worst of both
worlds. So I measured it directly: on the same time budget, how much does an honest ensemble
actually improve over a hyperparameter-tuned LightGBM? The % gain, and whether it's real.*

---

## The starting point (the same across the whole series)

The shared rig, briefly:

- Imbalanced binary datasets, metric **Average Precision** (PR-AUC). The series draws on 89 of
  them, but the ensemble experiments here are far heavier per run (an ensemble is ~5× a single
  model), so this part uses a **representative 16-dataset subset** — and **8** for the
  equal-time comparison, its most expensive piece.
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

And it's worth remembering *why* LightGBM is fast in the first place. That was its entire
reason for existing: the original paper (Ke et al., NeurIPS 2017) set out to match the
accuracy of XGBoost-style gradient boosting **but train far faster and lighter** — reporting
up to ~20× speedups — via two tricks, **GOSS** (subsample the low-gradient examples) and
**EFB** (bundle sparse, mutually-exclusive features so there's less to scan). LightGBM is,
by design, the *save-time* tool.

Which makes the habit doubly ironic: **nobody trains just one LightGBM.** They try dozens,
hundreds, a thousand configs, sweeping hyperparameters — and *that* is where the time goes.
We reach for the model built to save time, and then spend it all anyway. Once you add up all
those configs, you've spent as much compute as an ensemble would have cost… you just have
nothing better than a single tuned LightGBM to show for it. **It's the worst of both worlds:
you throw away LightGBM's only advantage (speed), and you never reach the depth a diverse
ensemble gives you.**

So the honest question isn't "single model vs ensemble". It's: **given the time you were
going to burn on all those configs anyway, what gives the best AP — and by how much?** That's
what I set out to estimate: the improvement of a proper ensemble over a hyperparameter-tuned
LightGBM, as a percentage, and whether it's statistically real.

## The ensemble

I took a pool of **10 diverse model families** (logistic regression, Gaussian NB, kNN,
LightGBM, HistGBM, XGBoost, ExtraTrees, Random Forest, CatBoost, MLP — one fixed config
each, *no* per-family tuning) and combined them with a **greedy Caruana blend**: a weighted
average of probabilities whose weights are chosen by hill-climbing. Two symmetric flavors,
mirroring the E1/E2 split from part one:

- **ens-E1:** learn the blend on a separate `val`. Trains each family once → cheap.
- **ens-E2:** learn the blend on the CV out-of-fold predictions — so it cross-validates all
  ten families (4 folds each) instead of training them once → several times more work.

The headline — and this is the number I actually wanted: **an honest ensemble beats a
hyperparameter-tuned LightGBM by about +0.01 AP, roughly +1.5% relative, winning ~70% of the
time.** It's clearly significant (p ranges from **2×10⁻⁸** in the broad comparison to **6×10⁻⁶**
at strictly equal time) — a real lift, not noise. So the ceiling of "more configs of one model"
sits about a percent-and-a-half below what a small, diverse blend reaches. The catch: the
ensemble costs several times the compute. ens-E1 (the cheap, held-out-val ensemble), at equal
time, doesn't beat E1.

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
The advantage isn't about time; it's **structural**: ens-E2 blends **ten genuinely different
model families**, each catching patterns the others miss, with weights learned honestly on
out-of-fold predictions. Throwing more configs at a single LightGBM, or more members at a
blend trained on a held-out slice, simply saturates — you can't out-tune your way to the
diversity of a real ensemble.

*[FIGURE 2 — fig_isofinal.png]*

*[TABLE 1 — table3_isotime.png]*

## Where the time actually goes

It's worth seeing the cost side directly, because it holds a twist. First, what each strategy
actually trains — this is the part people get wrong:

- **Plain E1 and plain E2 both sweep the *same* 18 LightGBM candidates** (6 hyperparameter
  settings × 3 imbalance-handling options). That grid search *is* the habit this article is
  about — "plain E1/E2" was never a single model. E1 scores each candidate on a held-out val
  (~18 quick fits); E2 scores each one with **4-fold cross-validation**, so those 18 configs
  turn into **≈ 18 × 4 ≈ 70 model trainings**.
- **The ensembles don't search configs at all.** They train a fixed pool of **10 diverse
  families, one config each** (no per-family tuning). ens-E1 trains each family **once — just
  10 trainings — and averages them**; ens-E2 cross-validates each family (4 folds) and blends
  on the out-of-fold predictions, so **≈ 10 × 4 ≈ 40 trainings**.

That reshuffles the cost ranking in a way that catches people off guard:

- **ens-E1 is the *cheapest* of all four — even cheaper than plain E1 or E2.** But not by a
  landslide, and the counts tell you why: ens-E1 trains **10 models** (then just averages them),
  while plain E1 sweeps **18 configs** — the same ballpark. That's why the two end up close
  (~19s vs ~37s) rather than orders apart: ens-E1 edges ahead only *because it skips the config
  search*, not because it does dramatically less work.
- **ens-E2 is the priciest, ~5× the base strategies — but not because it searches more.** It
  actually runs **fewer** trainings than plain E2 (~40 vs ~70); the cost comes from
  cross-validating a pool of **heavy learners** — CatBoost, MLP, XGBoost and random forests are
  each far slower to train than a single LightGBM. A fast model × many trainings still ends up
  cheaper than slow models × fewer. That's the real asymmetry.
- When over-funded to t\*, E1@t\* and E2@t\* spend it on more grid configs, and ens-E1@t\*
  spends it on more members — and it *still* loses. Cheap-and-plentiful can't buy the
  structural edge.

*[TABLE 2 — table3b_ens_time.png]*

**The takeaway:** the "many LightGBM configs is faster" argument is a false economy. Those
configs cost the same time as an ensemble — and at equal time, the ensemble wins.

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
