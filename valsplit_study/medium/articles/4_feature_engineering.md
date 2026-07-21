# Does speculative feature engineering hurt? 89 datasets say "mostly no — but it's still a bad deal"

*Part four of a series putting common ML habits to the test on 89 imbalanced datasets. This
one is my favorite, because the belief is so widespread and so comforting: "tree models
discard irrelevant features, so inventing a few variables never hurts — and once in a while
you stumble on a useful one." Free lottery ticket, right? I bought a lot of tickets.*

---

## The starting point (the same across the whole series)

If you've read the earlier parts, you know the rig; here it is once more, briefly:

- **89 imbalanced binary datasets**, metric **Average Precision** (PR-AUC).
- **A fixed, shared test set** touched exactly once.
- Base learner: **LightGBM** with early stopping; per-fold preprocessing, no leakage.
- **Every p-value is a paired t-test on the same splits.** In plain terms: I compare two
  strategies *case by case* on the exact same train/test partitions, then ask whether one
  systematically beats the other. Pairing on identical data cancels out the "some datasets are
  just easier" noise, so a small p-value means the winner won on merit — not because it drew a
  luckier split. (Rule of thumb: p = 0.01 means a gap this consistent would happen by chance
  about 1 time in 100.)

For this particular question I use the strongest single tree strategy — **E2 with
LightGBM** (cross-validation picks trees and config, refit on all dev). One deliberate
choice: I *avoid* the ensemble here. The whole point is to test whether **trees** shrug off
useless features, and an ensemble's non-tree members (logistic regression, kNN, MLP) do
suffer from noise — they'd confound the answer.

---

## The experiment: feed it growing amounts of nonsense

At 100% of each dataset's development pool, I add **random, meaningless features** and watch
the AP. The invented features are arithmetic mashups of random feature pairs —
`v_i/v_j`, `v_i·v_j`, `v_i−v_j`, `|v_i−v_j|`, `v_i+v_j` — plus pure noise, added in growing
amounts: **10%, 20%, … up to 100%** of the original feature count (at 100%, half the columns
are invented garbage).

The result is a nuanced "mostly true":

- **The tree does largely ignore the noise.** AP stays almost flat as you pile it on —
  LightGBM's feature selection is doing its job.
- **But "it never hurts" is too strong.** At 100% noise there's a **small, borderline drag**
  (ΔAP ≈ **−0.004**, p = **0.066**), driven mostly by a couple of sensitive datasets while
  the rest shrug it off.
- **There was no free lunch.** On average, the random features *never* improved things by
  luck. Not once did the lottery ticket pay out.

*[FIGURE 1 — fig_feateng.png]*

*[TABLE 1 — table4_feateng.png]*

## The part everyone forgets: the time bill

Here's the real reason to stop doing this. Even though the AP drag isn't statistically
significant, **the model takes noticeably longer to train for exactly zero gain** — more
columns, more work, same score. And that's the whole problem: training time is not free. In
a companion experiment (part three of this series) I show that at **equal wall-clock time**,
the compute you spend matters enormously — an honest ensemble beats "more of a single
model" precisely because it uses the time better.

Seen through that lens, speculative feature engineering is a **strictly worse use of
compute**: you pay in time and get nothing back. That wasted time could have bought you more
models, a bigger grid, or an ensemble — all of which *do* help.

**The takeaway:** LightGBM won't collapse under nonsense features, so the belief is *mostly*
right. But it's a bad deal on two counts: heavy noise drags AP a little, and — more
importantly — it burns training time for no return. Spend your effort, and your seconds, on
features that actually mean something.

**How it was done.** Model: E2 with LightGBM (ens-E2 avoided — its non-tree members suffer
from noise and would confound the answer). 8 datasets, 100% of dev (capped at 8,000 rows for
cost), 2 seeds. The invented features are random meaningless combinations of feature pairs
plus pure noise, added in growing amounts (10%…100% of the original feature count; at 100%
half the columns are invented). We report the paired ΔAP vs 0% noise.

---

## The moral

**Trees are robust to junk, but robustness isn't the same as free.** Adding speculative
features almost never helps, occasionally hurts a little, and always costs training time.
The comforting belief ("it can't hurt, might help") is wrong on the economics even when it's
roughly right on the accuracy. Real feature engineering — features grounded in the problem —
is worth every second. Random ones are worth none.

*This is the last of a four-part series. The earlier parts covered validation splits &
retraining, hyperparameter search, and ensembles. Code and all the numbers are in the study
repo; every p-value is a paired t-test on the same splits.*
