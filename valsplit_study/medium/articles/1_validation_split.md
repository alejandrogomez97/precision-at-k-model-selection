# One validation set, or two? Testing an old habit on 89 datasets

*A few years ago I wrote a short Medium post with a slightly heretical claim: you don't
need a separate held-out set to pick the number of trees and another one to pick the model
— cross-validation can do both at once. It was mostly a theoretical argument. This time I
wanted proof. So I built a rig to test it on 89 imbalanced datasets — and while I was at
it, tested a second habit that rides along with the first: retraining the final model on
all the data. This is the first of a four-part series putting common ML habits to the
test.*

---

## Where this comes from

Both this article and the older post it follows were born from something I kept running
into — in data scientists I knew, and in forum thread after forum thread. The reasoning
always went like this:

> "You've already used cross-validation to do the early stopping — to pick the number of
> trees. So now you *can't* reuse those out-of-fold predictions to choose between models:
> you've already fit a hyperparameter on that set. You need a *separate* validation set for
> the model choice."

And every time, it nagged at me. Because here's the thing: **choosing a model is basically
the same act as choosing a hyperparameter.** In both cases you're picking, out of many
candidates, the configuration — or the architecture — that best fits a given set. And that
means you overfit that set a little, whether the "candidate" is `num_leaves = 63` or
"XGBoost instead of a random forest." In fact, much of the time model selection *is* just
choosing between different configurations of models of the same family — which is mostly
choosing hyperparameters by another name.

So why does nobody blink at picking the number of trees on one set and *every other
hyperparameter* on another? Follow that logic to its end and you'd need a brand-new
validation set for **every single hyperparameter** you tune. That never made much sense to
me.

But — I'd seen it more than once, from people who clearly knew what they were doing. So I
kept wondering: is there something *special* about the number of trees, some effect from
early stopping I just wasn't seeing, that genuinely would demand that second held-out set?
There was only one way to settle it. I ran the experiment.

## The starting point (the same across the whole series)

Every article in this series is judged the same way, so here's the shared rig, once:

- **89 imbalanced binary datasets** (from imblearn and OpenML). Metric: **Average
  Precision** (PR-AUC), the right choice under class imbalance.
- **A fixed, shared test set** (30%, capped at 6,000 rows) that every strategy is judged
  on — touched exactly once.
- **A fraction sweep instead of absolute sizes.** The point is to see how each habit
  behaves **when you have lots of training data versus very little**. So instead of fixed
  "300, 600, 1200… rows", I vary the *fraction* of each dataset's development pool (10%,
  20%, …, 100%). Why fractions? Because datasets differ in size: sweeping by absolute count
  means only the big datasets reach the large sizes, and if those happen to be the hard
  ones your aggregate curve dips at the end and lies to you. With fractions, **every dataset
  is present at every point**. (I learned this the hard way — my first plots showed
  performance *dropping* with more data. Pure composition artifact.)
- Base learner: **LightGBM**, 18 grid candidates (6 configs × 3 imbalance techniques),
  number of trees always set by early stopping. Per-fold preprocessing, no leakage.
- **Every p-value below is a paired t-test on the same splits.** In plain terms: I compare
  the two strategies *case by case* on the exact same train/test partitions, then ask whether
  one systematically beats the other. Pairing on identical data cancels out the "some datasets
  are just easier" noise, so a small p-value means the winner won on merit — not because it
  drew a luckier split. (Rule of thumb: p = 0.01 means a gap this consistent would happen by
  chance about 1 time in 100.)

---

## The habit: one validation set, or two?

Here are the two contenders, both starting from the same development pool and judged on the
same test set:

- **E1 — separate validation.** Split dev into `train` + `val`. Cross-validate *within
  train* to fix the number of trees (early stopping). Then pick the best candidate on the
  independent `val` set. This is the "two sets" school.
- **E2 — CV only.** Run cross-validation over all of dev; the out-of-fold predictions decide
  **both** the number of trees **and** the winning candidate. This is the "let CV do
  everything" school — the one my old post defended.

Across **1,780 paired comparisons**, the verdict is sober but clear: **the two are almost
identical, with a small, systematic edge to E2** — mean ΔAP(E1−E2) = **−0.0037**,
**p = 9×10⁻⁷**. And the edge has a shape: with **scarce data** E2 wins clearly (≤30% of the
pool: −0.0049, **p = 0.006**); with **abundant data** it narrows to a tie (≥80%: −0.0018,
p = 0.09). The gap also grows with class imbalance — exactly where E1's tiny `val` set has
too few positives to choose well.

**The takeaway:** carving off a separate validation set buys you nothing, and quietly costs
you when data is scarce or imbalanced. My years-old claim survives contact with the data.
Use CV.

**How it was done.** E1: split dev into `train` (75%) + `val` (25%); 4-fold CV on `train`
fixes the number of trees (early stopping on Average Precision, up to 3,000 trees, patience
50); the 18 candidates (6 LightGBM configs — num_leaves ∈ {15,31,63} × reg_lambda ∈ {0,1} —
× 3 imbalance techniques: none / class_weight / SMOTE) are scored on `val`; the winner is
refit on `train`+`val`. E2: 4-fold CV over all of dev; the out-of-fold predictions pick both
trees and candidate; the winner is refit on dev. 89 datasets × 10 fractions × 2 seeds =
1,780 paired cells.

*[FIGURE 1 — fig_A_story.png]*

*[TABLE 1 — table1_e1_vs_e2.png]*

---

## The habit that rides along: "I always retrain on everything"

Once you've picked your model, do you refit it on *all* the data before shipping? Almost
everyone does, on reflex — "to be safe". But whether it helps turns out to **depend on how
you did the selection** — and it's a neat little example of what actually happens when you
keep some data aside and never train on it. I evaluated four variants — E1/E2 × with/without
a final retrain — on the same splits:

- **E1: yes, retrain.** ΔAP = **+0.0091**, **p = 2×10⁻³⁶**. E1 held out a `val` set it never
  trained on — retraining on `train`+`val` hands that data back. E1-without-retrain is the
  worst of the four.
- **E2: don't bother — it can even hurt.** ΔAP = **−0.0036**, **p = 1×10⁻¹⁰**. CV already
  used every example; the honest "no-retrain" here is a *bagged* ensemble of the fold
  models, and it slightly beats the refit.

**The takeaway:** retraining isn't a virtue in itself — it's a way to recover data you held
out. If you didn't hold any out (CV), there's nothing to recover.

**How it was done.** Same 18-candidate setup, evaluating all four combinations (E1/E2 ×
retrain/no-retrain) on identical splits. "No-retrain" means: for E1, keep the model trained
on `train` only; for E2, a *bagged* ensemble of the 4 fold models (no refit on dev). 89
datasets, paired t-test.

*[FIGURE 2 — fig_retrain.png]*

---

## The moral

Two habits, two verdicts that turn out to be the same idea seen twice:

1. **One validation set is enough** for imbalanced data — and marginally *better* than two.
   CV can pick both the trees and the model. *p = 9×10⁻⁷.*
2. **Retraining only pays if you held data out.** With a separate `val` set, refit and
   recover it (E1: p = 2×10⁻³⁶). With CV, you already used everything, so retraining does
   nothing — or slightly hurts (E2: p = 1×10⁻¹⁰).

Put together: **cross-validation is the honest default.** It uses all your data to choose,
leaves nothing to "recover", and doesn't make you babysit a fragile little validation set
that's especially untrustworthy when positives are scarce.

*Next in the series: does fancy hyperparameter search (Bayesian, evolutionary, Hyperband…)
actually beat a humble grid? Code and all the numbers are in the study repo.*
