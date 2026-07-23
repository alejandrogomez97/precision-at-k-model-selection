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
the AP. Each invented feature is one of six kinds, picked uniformly at random: five are
arithmetic mashups of a random pair of real features — `v_i/v_j`, `v_i·v_j`, `v_i−v_j`,
`|v_i−v_j|`, `v_i+v_j` — and the sixth is pure Gaussian noise. So **≈5/6 of the invented
columns are random combinations of real features and ≈1/6 is pure noise** — mimicking how
people actually do "feature engineering on a hunch" (mostly ratios/products/differences,
with the occasional meaningless column thrown in). I add them in growing
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
cost), 2 seeds. Each invented feature is one of six kinds chosen uniformly at random — five
arithmetic combinations of a random feature pair (`v_i/v_j`, `v_i·v_j`, `v_i−v_j`,
`|v_i−v_j|`, `v_i+v_j`) and one pure-Gaussian-noise column — so ≈5/6 are random combos and
≈1/6 is pure noise. Added in growing amounts (10%…100% of the original feature count; at 100%
half the columns are invented). We report the paired ΔAP vs 0% noise.

---

## What about *removing* features?

If *adding* variables doesn't help, maybe the opposite does: **removing redundant ones**.
Datasets often carry features that are near-copies of each other, and the folklore says
pruning them cleans up the model. So I ran the mirror experiment. For each dataset I computed
the **Spearman correlation between every pair of features** (on the development data only, in
absolute value — a strong negative correlation is just as redundant as a positive one), and
swept a threshold: at each level I dropped any feature that correlated **≥ threshold** with
another one I'd already kept. Same E2 + LightGBM setup, judged on the same test set.

The result is the mirror image of the "adding" story — **removing doesn't help either, and
overdoing it hurts a lot:**

- **Cutting near-duplicates is harmless (and a touch faster).** At `|ρ|≥0.99` and `|ρ|≥0.95`
  you drop ~9–14% of the features and AP doesn't move (ΔAP +0.003 and +0.001, not
  significant), while training shaves a few percent off the clock.
- **Cutting anything more is actively destructive.** At `|ρ|≥0.85–0.75` you remove ~40% of
  the columns and AP **collapses by −0.04 to −0.06** (p ≤ 0.04) — features that are "only"
  0.75–0.85 correlated still carry complementary signal, and throwing them out is throwing
  away accuracy.

*[TABLE 2 — table6_featreduce.png]*

To pin down exactly *where* removal stops being safe, I zoomed in with a fine sweep from
`|ρ|≥0.90` to `1.00` in steps of 0.005 (correlations in absolute value, so a −0.92 counts the
same as +0.92). The picture is flat: **across the entire 0.90–1.00 range, no threshold moves
AP significantly** (every paired *p* > 0.12). You can drop anywhere from ~9% (at 0.99) to
~24% (at 0.90) of the features and the curve just hugs the baseline. You might notice the
ΔAP turn slightly positive from `|ρ|≥0.95` onwards — a nominal +0.0008 rising to +0.0026 by
0.99, where only near-perfect duplicates get cut — but **it is not statistically significant**
(paired *p* ≈ 0.34–0.69; the ±1 SE band in the figure straddles zero the whole way). So no,
removing near-duplicates does not reliably improve AP; that bump is noise. The damage from the coarse sweep only kicks in
*below* 0.90 (by 0.85 it's a significant −0.043). In other words: **≈0.90 is the floor of the
safe zone**, and even inside it there's no real prize — removal is, at best, harmless.

*[FIGURE 2 — fig_featreduce_fine.png]*

One clarification, because it's easy to conflate: this is **not** what LightGBM's **EFB**
(Exclusive Feature Bundling, from Part 3) does. EFB bundles **sparse, mutually-exclusive**
features — ones that are almost never non-zero at the same time (think one-hot columns) — and
it's essentially lossless, done purely for speed. Our pruning is the opposite kind of move:
it deletes **densely correlated** features by hand, and it can — and does — throw away signal.
LightGBM already copes with redundancy internally (a correlated copy just rarely gets chosen
for a split); doing its job manually, on a hunch, mostly just risks hurting you.

**Takeaway:** removing features is like adding them — it won't raise your AP. If you only cut
*near-perfect* duplicates (`|ρ|≥0.95`) it's safe and trims a little time; anything more
aggressive quietly deletes accuracy.

---

## The moral

**Trees are robust to junk, but robustness isn't the same as free — and neither direction of
mindless feature fiddling pays off.** *Adding* speculative features almost never helps,
occasionally hurts a little, and always costs training time. *Removing* features by
correlation doesn't help either: safe only if you cut near-perfect duplicates, and
destructive (−0.04 to −0.06 AP) if you get greedy. Both point the same way: LightGBM already
handles redundant and useless columns better than your reflexes do. Real feature engineering
— features grounded in the problem — is worth every second. Inventing or pruning on a hunch
is worth none.

*This is the last of a four-part series. The earlier parts covered validation splits &
retraining, hyperparameter search, and ensembles. Code and all the numbers are in the study
repo; every p-value is a paired t-test on the same splits.*
