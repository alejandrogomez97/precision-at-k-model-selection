# Precision@100 Is a Great Target and a Dangerous Judge

### The metric you care about can be the wrong one to *train* on — and, more surprisingly, the wrong one to *select* on

Imagine you run the fraud unit at an electricity utility. Millions of meters are out there, some rigged. Your model scores every installation with a probability of fraud. And you have a crew that can inspect **100 installations a month**. Not 101, not 500. A hundred.

That single number changes everything. If a real fraudster ends up ranked 250th, it makes no difference whether your model put them at the 80th percentile or the 50th — they will never be inspected. The only thing that makes money is the **concentration of real fraud inside the top 100**. Everything else — beautiful calibration across the bottom 99%, a smooth ROC curve — is, operationally, decoration.

This is a **fixed-budget top-k problem**: you rank a population, act on the top *k*, and *k* is set from outside the model (a crew size, a budget, a legal quota). The natural score is **precision@k** — of the *k* things you acted on, what fraction were real?

<p class="eq"><span class="eq-lhs">precision@<em>k</em></span><span class="eq-eq">=</span><span class="frac"><span class="frac-num">true positives in the top <em>k</em></span><span class="frac-den"><em>k</em></span></span></p>

You see this shape everywhere once you look: tax audits (inspect the top 2,000 returns), retention campaigns (500 offers), medical call-backs (30 patients this week). Same skeleton: a queue, a team that reaches only the top of it, and a hard floor below which the model's opinion is irrelevant.

So the objective isn't "be good on average." It's "make the top 100 as clean as possible." Which raises the obvious question: if precision@100 is what I care about, why not train on it?

---

## Why precision@100 cannot be your training loss

Because it is **piecewise-constant and non-differentiable**, so its gradient is zero almost everywhere — and a zero gradient teaches a model nothing.

Precision@100 depends on your scores *only through which 100 items end up on top*. Nudge one prediction by a hair: almost always the top-100 set doesn't change, so precision@100 doesn't change, so its derivative is exactly **zero**. The metric only reacts at the instant an item crosses the cutoff and swaps with its neighbour — a sudden jump of 1/100. Flat, flat, flat, step, flat.

**Figure 1** shows it: slide one item's score and precision@k is a flat line with a single cliff, while log-loss is a smooth curve that always knows which way is better.

![Figure 1 — precision@k is piecewise-constant (gradient 0 almost everywhere); log-loss is smooth](fig1_step_function.png)

This is exactly why gradient boosting (and neural nets, and anything trained by gradient descent) uses log-loss. Boosting fits each new tree to the **negative gradient of the loss** — the "here's what you got wrong, and which way to fix it" signal. Make that concrete: eight installations, budget k = 3, mid-training.

| item | score *p* | label *y* | log-loss gradient (*p − y*) | precision@3 gradient |
|---|---|---|---|---|
| A | 0.92 | fraud (1) | −0.08 | 0 |
| B | 0.85 | legit (0) | **+0.85** | 0 |
| C | 0.80 | legit (0) | **+0.80** | 0 |
| D | 0.78 | fraud (1) | −0.22 | 0 |
| E | 0.60 | fraud (1) | −0.40 | 0 |
| F | 0.55 | legit (0) | +0.55 | 0 |
| G | 0.30 | fraud (1) | −0.70 | 0 |
| H | 0.10 | legit (0) | +0.10 | 0 |

The top three are A, B, C, so **precision@3 = 1/3** (only A is fraud). For log-loss with a sigmoid the gradient is simply ***p − y***: B and C are false positives sitting high, so they get big positive gradients ("push down"); D, E, G are missed frauds ("push up"). Every row gets a signed instruction.

Now precision@3. Nudge D's score up a hair (0.78 → 0.781): still fourth, top-3 unchanged, precision still 1/3 — derivative **0**. Do it for all eight rows and the last column is **zeros all the way down**. There *is* a helpful move — lift D above C, turning the top-3 into {A, B, D} and precision@3 into 2/3 — but that's a *finite* jump, and a gradient only sees the infinitesimal neighbourhood. So the tree is handed a target of all-zeros and learns nothing (Figure 2).

![Figure 2 — one boosting step: log-loss gives every row a direction; precision@3 gives zero](fig_gradient_example.png)

Notice the quiet punchline (right panel): log-loss's gradient is already pushing C *down* and D *up* — exactly the swap that would raise precision@3. The smooth surrogate, without being told about the budget, nudges toward the very move the budgeted metric wanted but couldn't ask for. (There are differentiable surrogates for top-k losses — Kar et al. 2015, Boyd et al.'s "Accuracy at the Top", the LambdaMART trick — but for most systems the pragmatic answer stays: **train on log-loss, evaluate on precision@100.**)

---

## The real question: should it even be your *selection* metric?

During training we need a derivative, so precision@100 is out. During **evaluation** — early stopping, and choosing which of hundreds of hyperparameter configs to ship — we only need to *compute* it. So it's allowed. And here almost everyone stops thinking: of course you select on the metric you care about.

But there are **two forces pulling in opposite directions.**

**Force 1 — alignment (for precision@100).** Log-loss is a *global* objective; it rewards calibration across the whole distribution, not the top 100. Suppose fraud comes in families: a common, easy one (*blatant meter tampering*) the model already scores high, and rarer, subtler ones (*quiet bypasses*) sitting mid-ranking. To lower average log-loss the model can profitably nudge many subtle cases from 0.3 to 0.5 — a big gain over many rows — even if it lets a few blatant frauds slip from rank 90 to 110, out of the budget. Log-loss improves; precision@100 — the money — drops. **A model that looks better by log-loss can be worse at the only thing that pays**, and only precision@100 would catch it.

**Force 2 — stability (against precision@100).** Precision@100 reads only 100 rows — really, only how many of those are positive. It's a proportion from a tiny sample, so it's *noisy*. Pick the best of hundreds of configs on the same validation set and you partly pick whichever one **got lucky** — the *winner's curse* — so the choice generalizes worse, and the reported number is inflated.

Which force wins is not an armchair question. So I measured it.

### How I measured it

The design is simple, and worth stating plainly because the result hinges on it:

1. Train a bank of **40 models** (random hyperparameters, all on log-loss) on a `train` split.
2. On a `validation` split, score every model with each candidate metric: **precision@k, average precision (AP), AUC, log-loss**.
3. For each metric, keep the model that **maximises it on validation** — that's "selecting by that metric."
4. Measure that chosen model's **precision@k on a held-out `test` split** — the real objective, identical for every metric.
5. Ask: whose pick — precision@k's or a rival's — has the higher *test* precision@k?

I ran this over **93 datasets** (90 real imbalanced ones from the `imbalanced-learn` and OpenML collections — fraud, credit default, churn, bankruptcy, medical screening… — plus 3 synthetic), sweeping the budget K across a wide range per dataset. To pool wildly different datasets I use **normalized regret**: 0 = you picked the best model in the bank, 1 = no better than an average one. Lower is better.

The headline is the counter-intuitive part:

> **The model chosen by log-loss (or AP) on validation usually has a *higher* test precision@k than the one chosen by validation precision@k itself.**

Selecting on the exact metric you care about gives you a worse result on that metric — because you're partly selecting on validation luck (Force 2), and log-loss's more stable pick lands on a genuinely better model.

---

## What 93 datasets say

**1. On average, log-loss and average precision win; precision@k trails** (Figure 3). Mean regret: log-loss 0.71, AP 0.72, precision@k 0.79 (AUC in between). By how often a metric picks the single best model: log-loss 34%, precision@k 25%, AP 21%, AUC 20%.

![Figure 3 — model-selection regret across 93 datasets](figT0_overall.png)

**2. It's not a ratio — it's a count.** The natural instinct is that precision@k should be fine whenever the budget is "big enough" relative to the positives — some scale-free ratio like *n_pos / K*. It isn't: that ratio barely predicts anything. What predicts selection difficulty is the **absolute number of true positives that land in the budget** — the more positives up there, the better *every* metric selects (Figure 4, right). And it's a far stronger relationship than the ratio's: a Spearman correlation of **0.43** in magnitude, versus a mere **0.19** for the ratio. The reason is basic statistics — precision@k is a proportion, and a proportion's reliability is set by the number of successes behind it, a *count*, not a fraction. Two shops with the same ratio behave differently if one is ten times larger.

![Figure 4 — the ratio n_pos/K barely orders regret; the absolute count of positives in the budget does](figT2_ratio_vs_count.png)

**3. And you can know that count in advance.** Fair objection: "positives in the budget" ≈ precision × K depends on the model — the thing you're selecting. But its **model-free ceiling, `min(K, n_pos)`** — the most positives that *could* land in the budget, known from your budget and how many positives you have — predicts almost as well (a correlation of 0.38, nearly the 0.43 above) and tracks the true count almost perfectly (0.94). So the deciding quantity is knowable before you train (Figure 5). Want the exact number? Train one cheap baseline and count the true positives in its top-k on validation.

![Figure 5 — the a-priori min(K, n_pos) tracks the endogenous precision × K almost perfectly](figT6_apriori.png)

There's **no clean regime where precision@k rules**: it only edges ahead in the corner where the budget is tiny relative to a large positive pool, and never by much. And the flip side of its noise: reported validation precision@k **overstates** true precision the fewer positives sit in the budget — by tens of percent, occasionally more than 2× — so it also hands you a rosier number than production will show.

---

## Practical takeaways

- **Keep precision@k as your target and headline metric — never as your training loss.** Train on log-loss; report precision@budget because it's what the business feels.
- **To choose which model to ship, default to average precision or log-loss.** Across 93 datasets they tied for best and won most often; precision@k was measurably worse and occasionally blew up.
- **Don't look for a ratio threshold — there isn't one.** The lever that matters is `min(budget, positives you have)`: the count of positives that can land in your budget. If it's small, *no* selection metric will pick reliably — get more labelled positives or a bigger budget, not a cleverer metric.
- **Use precision@k as a tie-breaker, not a primary selector**; when it disagrees with a stable metric, trust the stable one. Skip AUC for fixed-budget problems.
- **Never quote raw validation precision@k as expected production performance** when the budget is positive-poor — report a nested / held-out estimate.

A compact mental model: precision@k is a proportion measured from the positives inside your budget, so its reliability — as a *selector* and as a *reported number* — is set by **how many positives that is**, not by any ratio of rates. Its alignment is the reason to want it; its variance is the reason to distrust it; grow the count, don't tune a ratio.

*Limitations: one model family (LightGBM), 40 configs per dataset, selection studied on a fixed bank (early stopping should behave the same). "True quality" on small datasets is itself noisy, mitigated by pooling 93 datasets. Every recommendation is about expected quality — on one project you can still get lucky. Code and figures are reproducible from a fixed seed (`experiment_real3.py`, `meta_real3.py`).*
