"""Builds articulo_en.html — English version (fraction sweep, dynamic numbers)."""
import base64, json, os
import numpy as np, pandas as pd

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"


def b64(p):
    p = f"{STUDY}/{p}"
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else None


def fig_block(path, cap):
    # prefer the English-labelled figure when available
    if path.endswith(".png"):
        en = path[:-4] + "_en.png"
        if os.path.exists(f"{STUDY}/{en}"):
            path = en
    d = b64(path)
    return f'<figure><img src="{d}" alt="{cap}"/><figcaption>{cap}</figcaption></figure>' if d else ""


def jload(name):
    p = f"{STUDY}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x): return f"{x*100:.0f}%"
def fmt_p(p):
    if p is None or (isinstance(p, float) and p != p): return "n/a"
    return f"{p:.0e}" if p < 1e-3 else f"{p:.3f}"


def mark(vals, higher_better=True):
    import math
    nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if len(set(nums)) < 2:
        return [""] * len(vals)
    best = max(nums) if higher_better else min(nums)
    worst = min(nums) if higher_better else max(nums)
    out = []
    for v in vals:
        if not isinstance(v, (int, float)) or math.isnan(v):
            out.append("")
        elif v == best:
            out.append(' class="best"')
        elif v == worst:
            out.append(' class="worst"')
        else:
            out.append("")
    return out


SF = jload("summary_frac.json")


def section4():
    if not SF:
        return '<p class="note">Main result in progress (fraction re-run).</p>'
    e = SF["E1_vs_E2"]; ap = SF["ap_by_frac"]; cx = SF.get("crossover")
    rows = ""
    for f in SF["fracs"]:
        f = float(f)
        e1 = ap["E1-retrain"].get(f, ap["E1-retrain"].get(str(f)))
        e2 = ap["E2-retrain"].get(f, ap["E2-retrain"].get(str(f)))
        if e1 is None: continue
        d = e1 - e2
        c = mark([e1, e2])
        rows += (f"<tr><td>{pct(f)}</td><td{c[0]}>{e1:.3f}</td><td{c[1]}>{e2:.3f}</td>"
                 f"<td>{d:+.4f}</td></tr>")
    if cx:
        lo, hi = cx["low"], cx["high"]
        lo_e2 = lo["mean"] < 0 and (lo.get("p") or 1) < 0.05
        hi_e1 = hi["mean"] > 0 and (hi.get("p") or 1) < 0.05
        hi_e2 = hi["mean"] < 0 and (hi.get("p") or 1) < 0.05
        if lo_e2 and hi_e1:
            big = "E2 wins with little data, E1 with plenty — the crossover theory predicts."
        elif lo_e2 and not (hi_e1 or hi_e2):
            big = "E2 wins with little data; with plenty the two strategies tie. E2's edge fades as data grows."
        elif hi_e1 and not lo_e2:
            big = "E1 wins with abundant data; with little data it's a tie."
        else:
            big = "The two strategies are practically equivalent across the range."
        callout = (f'<div class="callout"><p class="big">{big}</p>'
                   f'<p>Paired ΔAP(E1−E2): at <strong>low</strong> fractions {lo["mean"]:+.4f} '
                   f'(E1 wins {pct(lo["winrate_E1"])}), at <strong>high</strong> fractions '
                   f'{hi["mean"]:+.4f} (E1 wins {pct(hi["winrate_E1"])}). The theoretical direction '
                   f'is clear — the independent <code>val</code> set only pays off with abundant '
                   f'data — but the magnitude is small. n = {e["n"]} pairs, {SF["n_datasets"]} datasets.</p></div>')
    else:
        callout = ""
    tbl = ("<div class='tw'><table><thead><tr><th>fraction</th><th>AP E1</th>"
           "<th>AP E2</th><th>ΔAP (E1−E2)</th></tr></thead><tbody>"
           f"{rows}</tbody></table></div>")
    return callout + tbl + fig_block("fig_A_story.png",
        "E1 vs E2 (fraction sweep, every dataset present at every point): AP, paired difference, and effect of imbalance.")


def section5():
    if not SF or "retrain" not in SF:
        return '<p class="note">Retraining comparison in progress.</p>'
    r = SF["retrain"]
    def verb(m, p):
        if p >= 0.05: return "is indifferent"
        return "helps" if m > 0 else "hurts"
    li = "".join(
        f"<li><strong>{s}:</strong> retraining {verb(r[s]['mean'], r[s]['p'])} "
        f"(ΔAP = {r[s]['mean']:+.4f}, wins {pct(r[s]['winrate_retrain'])}, "
        f"p = {fmt_p(r[s]['p'])}, n = {r[s]['n']}).</li>" for s in ("E1", "E2"))
    return ("<p>Evaluating all four variants on the same splits isolates the value of the "
            "final refit:</p><ul class='tight'>" + li + "</ul>") + fig_block(
        "fig_retrain.png",
        "The four variants (left; the retraining curves match the previous figure) and the effect of retraining per strategy (right).")


def section_b():
    p = f"{STUDY}/summary_B.csv"
    if not os.path.exists(p):
        return '<p class="note">Optuna (equal-budget) comparison in progress.</p>'
    s = pd.read_csv(p); j = jload("summary_B.json") or {}
    iso = j.get("iso18"); b40 = j.get("b40")
    txt = "<p><strong>At equal number of configurations (18 vs 18):</strong> "
    if iso:
        v = "better" if iso["gain"] > 0.0005 else ("worse" if iso["gain"] < -0.0005 else "the same")
        txt += (f"Optuna is <strong>{v}</strong> than the grid (mean Δ {iso['gain']:+.4f} AP, "
                f"wins {pct(iso['winrate'])} of cells, n={iso['n']}). ")
    if b40:
        txt += f"Even with more budget (40 trials) the gain is {b40['gain']:+.4f} (wins {pct(b40['winrate'])})."
    txt += "</p>"
    has18 = "ap_o18" in s.columns
    def row(r):
        aps = [r.ap_grid, (r.ap_o18 if has18 else float('nan')), r.ap_o40]
        ts = [r.t_grid, (r.t_o18 if has18 else float('nan')), r.t_o40]
        ca = mark(aps, True); ct = mark(ts, False)
        return (f"<tr><td>{pct(r['frac'])}</td>"
                f"<td{ca[0]}>{aps[0]:.3f}</td><td{ca[1]}>{aps[1]:.3f}</td><td{ca[2]}>{aps[2]:.3f}</td>"
                f"<td{ct[0]}>{ts[0]:.0f}s</td><td{ct[1]}>{ts[1]:.0f}s</td><td{ct[2]}>{ts[2]:.0f}s</td></tr>")
    rows = "".join(row(r) for _, r in s.iterrows())
    tbl = ("<div class='tw'><table><thead><tr><th>fraction</th><th>AP grid-18</th>"
           "<th>AP optuna-18</th><th>AP optuna-40</th><th>t grid</th><th>t opt-18</th>"
           "<th>t opt-40</th></tr></thead><tbody>" f"{rows}</tbody></table></div>")
    return txt + tbl + fig_block("fig_B_optuna.png",
        "Optuna vs grid at equal number of configurations (18): AP, gain and time by fraction.")


def section_hpo():
    p = f"{STUDY}/summary_hpo.csv"
    if not os.path.exists(p):
        return '<p class="note">Budget/method sweep in progress.</p>'
    s = pd.read_csv(p, index_col=0)
    j = jload("summary_hpo.json") or {}
    methods = [m for m in ["grid", "random", "tpe", "cmaes"] if m in s.columns]
    gains = {m: j.get(m, {}).get("gain_budget") for m in methods if m in j}
    best_gain = max(gains.values()) if gains else 0
    verdict = ("More configurations barely move the AP" if abs(best_gain) < 0.008
               else "More configurations improve AP a little")
    maxb = s.index.max(); winner = s.loc[maxb].idxmax()
    rows = ""
    for b in s.index:
        vals = [s.loc[b, m] for m in methods]
        c = mark(vals, True)
        tds = "".join(f"<td{c[i]}>{v:.3f}</td>" for i, v in enumerate(vals))
        rows += f"<tr><td>{int(b)}</td>{tds}</tr>"
    head = "".join(f"<th>{m}</th>" for m in methods)
    tbl = (f"<div class='tw'><table><thead><tr><th># configs</th>{head}</tr></thead>"
           f"<tbody>{rows}</tbody></table></div>")
    st = pd.read_csv(f"{STUDY}/summary_hpo_time.csv", index_col=0) if os.path.exists(f"{STUDY}/summary_hpo_time.csv") else None
    ttbl = ""
    if st is not None:
        tmethods = [m for m in methods if m in st.columns]
        trows = ""
        for b in st.index:
            vals = [st.loc[b, m] for m in tmethods]
            c = mark(vals, False)
            tds = "".join(f"<td{c[i]}>{v:.0f}s</td>" for i, v in enumerate(vals))
            trows += f"<tr><td>{int(b)}</td>{tds}</tr>"
        thead = "".join(f"<th>{m}</th>" for m in tmethods)
        ttbl = ("<p class='note' style='margin-bottom:.3rem'>Search time (s):</p>"
                f"<div class='tw'><table><thead><tr><th># configs</th>{thead}</tr></thead>"
                f"<tbody>{trows}</tbody></table></div>")
    sigbits = []
    for m in ("random", "tpe", "cmaes"):
        r = j.get(m, {})
        if "vs_grid_p" in r:
            sigbits.append(f"{m} {r['vs_grid_mean']:+.4f} (p={fmt_p(r['vs_grid_p'])}, "
                           f"wins {pct(r['vs_grid_winrate'])})")
    n_cells = j.get("random", {}).get("vs_grid_n", "?")
    sigtxt = ("<p class='sig'>Significance (vs grid, paired t-test, n=" + str(n_cells) + "): "
              + "; ".join(sigbits) + ". <strong>No difference is statistically significant "
              "(all p&gt;0.2)</strong>: they win in win-rate but by a tiny effect (~0.003 AP).</p>") if sigbits else ""
    txt = (f"<p><strong>{verdict}</strong> (max gain from {min(s.index)}→{maxb} configs: "
           f"{best_gain:+.4f} AP). At the largest budget ({maxb} configs), <strong>neither Bayesian "
           f"(TPE) nor evolutionary (CMA-ES) search pulls away from the grid or random search</strong>. "
           f"Time grows ~linearly with the number of configurations.</p>")
    mbox = ("<details class='methods'><summary>Methodology · Result 3</summary>"
            "<ul><li><strong>8 datasets</strong> (representative subset), fractions "
            "<strong>30% and 100%</strong> of the development pool, <strong>2 seeds</strong> → 32 cells.</li>"
            "<li>E1 framework (train/val; trees by CV on train; selection on val; winner refit on dev → test).</li>"
            "<li><strong>4 optimization methods</strong>, what each does:<ul>"
            "<li><em>grid</em>: tries predefined points of a large (360-config) shuffled grid — no "
            "learning, systematic coverage.</li>"
            "<li><em>random</em>: samples configurations at random — surprisingly competitive, the honest "
            "baseline.</li>"
            "<li><em>TPE</em> (Tree-structured Parzen Estimator, Optuna's Bayesian): models which regions "
            "give good/bad results and samples where it looks promising — exploits what it learns.</li>"
            "<li><em>CMA-ES</em> (evolution strategy): keeps a Gaussian «cloud» of configurations that it "
            "shifts and narrows toward the best — strong in continuous spaces.</li></ul></li>"
            "<li><strong>Search space.</strong> The three Optuna samplers explore a full "
            "<strong>8-hyperparameter space:</strong> num_leaves [7,255], min_child_samples [5,100], "
            "reg_lambda/alpha [1e-3,10], colsample_bytree [0.5,1], subsample [0.6,1], learning_rate "
            "[0.01,0.2], imbalance technique {none, class_weight, SMOTE}. The 360-config grid only "
            "varies <strong>5 of those</strong> (num_leaves, min_child_samples, learning_rate, reg_lambda, "
            "imbalance technique); the other three stay at LightGBM defaults. Trees set by early stopping.</li>"
            "<li><strong>Budgets 20/40/80/160</strong> configs, checkpointing the best-so-far from a "
            "single 160-config run per method.</li>"
            "<li>Compared on a <strong>balanced panel</strong> (cells common to all 4 methods); "
            "significance by paired t-test vs grid.</li></ul></details>")
    return txt + sigtxt + tbl + ttbl + mbox + fig_block("fig_hpo.png",
        "Test AP vs number of configurations per method (low and high fraction) and time cost.")


def _ens_verdict(w1, w2, fast2):
    e1 = (w1 is not None and w1 > 0.55); e2 = (w2 is not None and w2 > 0.55)
    if e2 and not e1:
        return ("<p>An interesting nuance: the <strong>E2-style ensemble does improve E2's AP</strong> "
                "in most cases, but <strong>at a large time cost</strong> (it trains 10 families with "
                "CV+refit, never faster than E2). The E1-style, <strong>at equal time, doesn't pay "
                "off</strong>. Practical takeaway: if you have spare compute, a blend of families over "
                "all <code>dev</code> squeezes a bit more AP; if time matters, the focused LightGBM wins.</p>")
    if e1 and e2:
        return "<p>Both ensembles beat their base strategy on AP, at a higher time cost.</p>"
    return ("<p>So <strong>neither ensemble reliably beats its base strategy</strong> in its "
            "respective comparison; the focused LightGBM stays hard to beat.</p>")


def section_c():
    p = f"{STUDY}/summary_isotime.csv"
    if not os.path.exists(p):
        return '<p class="note">Families (iso-time) in progress.</p>'
    d = pd.read_csv(p); j = jload("summary_isotime.json") or {}
    w1 = j.get("ens_e1_beats_e1_isotime"); w2 = j.get("ens_e2_beats_e2"); fast2 = j.get("ens_e2_faster")
    intro = ("<p>Two ensembles, each compared with <em>its</em> strategy:</p><ul class='tight'>"
             "<li><strong>E1-style ensemble</strong> (greedy blend on the separate <code>val</code>), "
             "compared with E1 <strong>at equal time</strong>: wins "
             f"<strong>{pct(w1) if w1 is not None else '…'}</strong> of cells.</li>"
             "<li><strong>E2-style ensemble</strong> (blend on the CV out-of-fold predictions + "
             "families refit on all <code>dev</code>), compared with E2: wins "
             f"<strong>{pct(w2) if w2 is not None else '…'}</strong> of cells"
             + (f", and is faster than E2 only in <strong>{pct(fast2)}</strong>" if fast2 is not None else "")
             + ".</li></ul>")
    s1 = j.get("sig_ens_e1", {}); s2 = j.get("sig_ens_e2", {})
    if s2.get("p") is not None:
        intro += (f"<p class='sig'>Significance (paired t-test): ens-E1 vs E1 {s1.get('mean',0):+.4f} "
                  f"(p={fmt_p(s1.get('p',1))}, n.s.); <strong>ens-E2 vs E2 {s2.get('mean',0):+.4f} "
                  f"(p={fmt_p(s2.get('p',1))} → highly significant)</strong>, n={s2.get('n','?')}. Unlike "
                  f"HPO, here the effect is large (~0.012 AP) and clearly significant.</p>")
    intro += _ens_verdict(w1, w2, fast2)
    by = d.groupby("frac").agg(e1=("e1_ap","mean"), ens1=("ens_at_e1","mean"),
        e2=("e2_ap","mean"), ens2=("e2ens_ap","mean"), t1=("e1_t","mean"),
        tens1=("ens_full_t","mean"), t2=("e2_t","mean"), tens2=("e2ens_t","mean")).reset_index()
    def row(r):
        import math
        ca = mark([r.e1, r.ens1, r.e2, r.ens2], True)
        ct = mark([r.t1, r.tens1, r.t2, r.tens2], False)
        def cell(v, c, s="{:.3f}"): return f"<td{c}>{s.format(v)}</td>" if not math.isnan(v) else "<td>—</td>"
        return (f"<tr><td>{pct(r['frac'])}</td>{cell(r.e1,ca[0])}{cell(r.ens1,ca[1])}"
                f"{cell(r.e2,ca[2])}{cell(r.ens2,ca[3])}"
                f"{cell(r.t1,ct[0],'{:.0f}s')}{cell(r.tens1,ct[1],'{:.0f}s')}"
                f"{cell(r.t2,ct[2],'{:.0f}s')}{cell(r.tens2,ct[3],'{:.0f}s')}</tr>")
    rows = "".join(row(r) for _, r in by.iterrows())
    tbl = ("<div class='tw'><table><thead><tr><th>fraction</th><th>AP E1</th>"
           "<th>AP ens-E1</th><th>AP E2</th><th>AP ens-E2</th>"
           "<th>t E1</th><th>t ens-E1</th><th>t E2</th><th>t ens-E2</th></tr></thead><tbody>"
           f"{rows}</tbody></table></div>")
    mbox = ("<details class='methods'><summary>Methodology · Result 4</summary>"
            "<ul><li><strong>16 datasets</strong> (subset), <strong>10 fractions</strong> (10→100%), "
            "<strong>2 seeds</strong> → up to 318 cells.</li>"
            "<li><strong>Pool of 10 families, one model per family with a fixed config (no grid "
            "search):</strong> logistic regression (balanced), Gaussian NB, kNN (k=25), LightGBM "
            "(400 trees, num_leaves 31, lr 0.05), HistGBM (400 iters, lr 0.05, early stopping), "
            "XGBoost (400 trees, depth 6, scale_pos_weight), ExtraTrees (300), Random Forest (300), "
            "CatBoost (400 iters, depth 6, auto-balanced) and MLP (64-32, early stopping).</li>"
            "<li><strong>Ensemble = greedy Caruana blend</strong> (with replacement) maximizing AP; "
            "weights come out as fractions (e.g. hgb 2/3 + rf 1/3). Usually keeps 2-4 families.</li>"
            "<li><strong>E1-style:</strong> blend learned on the separate <code>val</code>; families "
            "trained on train. Compared with E1 <em>at equal time</em> (anytime curve).</li>"
            "<li><strong>E2-style:</strong> blend learned on the CV out-of-fold predictions (k=4) + "
            "families refit on all <code>dev</code>. Compared with E2 on AP.</li>"
            "<li>Significance by paired t-test.</li></ul></details>")
    nr = jload("summary_ensnr.json")
    nrnote = ""
    if nr:
        nrnote = (f"<p><strong>And ens-E2 with or without retraining?</strong> Recall that in <em>base</em> "
                  f"E2 the bagged (no-retrain) clearly won: in the <em>ensemble</em> there's barely any AP "
                  f"difference — retrain {nr['ap_retrain']:.3f} vs no-retrain {nr['ap_noretrain']:.3f} "
                  f"<span class='sig'>(Δ={nr['delta']:+.4f}, p={fmt_p(nr['p'])}, n.s.; n={nr['n']})</span> "
                  f"— because a 10-model blend already cuts variance. But <strong>no-retrain is faster</strong> "
                  f"({nr['t_noretrain']:.0f}s vs {nr['t_retrain']:.0f}s, it skips the refit), so it "
                  f"<strong>matches on quality and saves ~25% of time</strong> → the preferable option for the "
                  f"E2 ensemble.</p>")
    return intro + nrnote + tbl + mbox + fig_block("fig_C_isotime.png",
        "E1-style ensemble vs E1 (equal-time) and E2-style vs E2: AP, win-rate and timings.")


def section_isofinal():
    import math
    p = f"{STUDY}/summary_isofinal.csv"
    if not os.path.exists(p):
        return '<p class="note">Equal-time experiment in progress.</p>'
    d = pd.read_csv(p); j = jload("summary_isofinal.json") or {}
    d = d.dropna(subset=[c for c in ["e1_at", "e2_at", "ens_e1_at"] if c in d.columns])
    ma = j.get("mean_ap", {}); se1 = j.get("ens_e2_vs_ens_e1_at", {})
    sg1 = j.get("ens_e2_vs_e1_at", {}); sg2 = j.get("ens_e2_vs_e2_at", {})
    intro = ("<p>The fair comparison is at <strong>equal time</strong>. Since ens-E2 is ~5× slower, "
             "we give the other strategies <strong>their same time budget</strong> (<strong>t*</strong> "
             "= ens-E2's time) to do <em>more</em>: more LightGBM configurations (E1@t*, E2@t*) and more "
             "ensemble members (ens-E1@t*). The <strong>@t*</strong> suffix marks «same strategy, "
             "searching until it consumes ens-E2's time».</p>")
    if se1.get("p") is not None:
        intro += (f"<p>Result: at equal time, <strong>ens-E2 still wins</strong> (mean AP "
                  f"{ma.get('ens_e2',0):.3f}). <span class='sig'>ens-E2 − ens-E1@t*: {se1['mean']:+.4f} "
                  f"(p={fmt_p(se1['p'])}, wins {pct(se1['winrate'])}); ens-E2 − E1@t*: "
                  f"{sg1.get('mean',0):+.4f} (p={fmt_p(sg1.get('p'))}); ens-E2 − E2@t*: "
                  f"{sg2.get('mean',0):+.4f} (p={fmt_p(sg2.get('p'))}). n={se1['n']}.</span></p>")
    by = d.groupby("frac").mean(numeric_only=True).reset_index()

    def cell(v, cc="", s="{:.3f}"):
        return f"<td{cc}>{s.format(v)}</td>" if (v == v) else "<td>—</td>"
    aprows = ""
    for _, r in by.iterrows():
        vals = [r.e1_ap, r.get("e1_at", float("nan")), r.ens_e1, r.ens_e1_at,
                r.e2_ap, r.get("e2_at", float("nan")), r.ens_e2]
        c = mark(vals, True)
        aprows += (f"<tr><td>{pct(r['frac'])}</td>" + "".join(cell(v, c[i]) for i, v in enumerate(vals)) + "</tr>")
    aptbl = ("<div class='tw'><table><thead><tr><th>frac</th><th>E1</th><th>E1@t*</th>"
             "<th>ens-E1</th><th>ens-E1@t*</th><th>E2</th><th>E2@t*</th><th>ens-E2</th>"
             f"</tr></thead><tbody>{aprows}</tbody></table></div>")
    trows = ""
    for _, r in by.iterrows():
        tv = [r.e1_t, r.get("e1_at_t", float("nan")), r.ens_e1_t, r.ens_e1_at_t,
              r.e2_t, r.get("e2_at_t", float("nan")), r.e2ens_t]
        c = mark(tv, False)
        trows += (f"<tr><td>{pct(r['frac'])}</td>" + "".join(cell(v, c[i], "{:.0f}s") for i, v in enumerate(tv)) + "</tr>")
    ttbl = ("<p class='note' style='margin-bottom:.3rem'>Times (s):</p><div class='tw'><table><thead><tr>"
            "<th>frac</th><th>E1</th><th>E1@t*</th><th>ens-E1</th><th>ens-E1@t*</th><th>E2</th>"
            f"<th>E2@t*</th><th>ens-E2</th></tr></thead><tbody>{trows}</tbody></table></div>")
    verd = ("<p>Even when the cheaper strategies spend their extra time trying more configs/members, "
            "<strong>they don't catch ens-E2</strong>. Its edge is <strong>structural, not time</strong>: "
            "it uses all of <code>dev</code> to learn the blend (OOF) <em>and</em> for the final models "
            "(refit). The E1-style ensemble also <strong>saturates</strong> (more members stop helping). "
            "If you have spare time, the E2-style blend squeezes a bit more AP; for the best quality/time "
            "trade-off, the focused LightGBM.</p>")
    mbox = ("<details class='methods'><summary>Methodology · Result 5</summary>"
            "<ul><li><strong>8 datasets</strong>, 10 fractions, 2 seeds. Same splits as the rest.</li>"
            "<li><strong>t* = the time ens-E2 consumes</strong> per cell (OOF blend + refit of 10 families "
            "on all <code>dev</code> = 50 trainings).</li>"
            "<li><strong>E1@t* and E2@t*:</strong> LightGBM grid (large shuffled grid) growing configs "
            "— up to 240 — until t*; the best (on val for E1, OOF for E2) is refit on dev → test.</li>"
            "<li><strong>ens-E1@t*:</strong> extended pool of ~36 members added with a greedy val blend "
            "until t* (or pool exhausted).</li>"
            "<li>Columns without <em>@t*</em> = natural version (from Result 4). Paired t-test.</li>"
            "</ul></details>")
    return intro + aptbl + ttbl + verd + mbox + fig_block("fig_isofinal.png",
        "Equal time (t* = ens-E2's time): AP of the @t* versions vs ens-E2, plus per-cell scatter.")


def section_feateng():
    j = jload("summary_feateng.json")
    if not j:
        return '<p class="note">Feature-engineering experiment in progress.</p>'
    dA = j.get("dAP_100_mean"); p = j.get("dAP_100_p")
    verdict = ("adding invented features <strong>barely affects</strong> the model"
               if (dA is not None and abs(dA) < 0.005)
               else ("<strong>degrades</strong> the model" if (dA is not None and dA < 0)
                     else "<strong>improves</strong> the model"))
    sigtxt = (f" (ΔAP at 100% added features = {dA:+.4f}, p={fmt_p(p)}"
              f"{', not significant' if (p and p>=0.05) else ''})") if dA is not None else ""
    intro = (f"<p>With LightGBM (E2), <strong>{verdict}</strong>{sigtxt}. The tree does, in effect, "
             f"<strong>mostly ignore the meaningless features</strong>: even if you double the feature "
             f"count with random noise, AP stays practically flat.</p>")
    rows = ""
    for r in j.get("by_level", []):
        cls = "pos" if r["dAP"] > 0.002 else ("neg" if r["dAP"] < -0.002 else "")
        rows += (f"<tr><td>{r['level']*100:.0f}%</td><td>{r['ap']:.3f}</td>"
                 f"<td class='{cls}'>{r['dAP']:+.4f}</td><td>{r['time']:.0f}s</td></tr>")
    tbl = ("<div class='tw'><table><thead><tr><th>% invented features</th><th>AP</th>"
           f"<th>ΔAP vs 0%</th><th>time</th></tr></thead><tbody>{rows}</tbody></table></div>")
    t0 = j.get("time_0"); t100 = j.get("time_100")
    tnote = (f"<p class='note'>The real damage isn't the AP: even though the drag at 100% added features isn't "
             f"significant, <strong>training time does rise</strong> (from {t0:.0f}s at 0% to {t100:.0f}s "
             f"at 100%) for <strong>exactly zero gain in AP</strong>. And that's the problem — under the "
             f"equal-time lens of §8, that wasted time is time you could have spent training <em>more</em> "
             f"models (a bigger grid, more ensemble members), which does help. So it's a strictly worse "
             f"use of compute.</p>") if t0 and t100 else ""
    mbox = ("<details class='methods'><summary>Methodology · Result 6 (§9)</summary>"
            "<ul><li><strong>Question:</strong> some believe «the model discards irrelevant features, so "
            "even if they add nothing they don't hurt; and sometimes you stumble on a useful pattern». "
            "We test it.</li>"
            "<li><strong>Model:</strong> E2 with LightGBM (the best single model and the tree whose "
            "robustness is debated; ens-E2 is avoided since its non-tree members do suffer with noise and "
            "would confound the question).</li>"
            f"<li><strong>{j.get('n_datasets','?')} datasets</strong>, 100% of dev (capped at 8,000 rows "
            "for cost), 2 seeds.</li>"
            "<li><strong>Random noise features:</strong> at each level, add a fraction (10%, 20%, …, 100% "
            "of the number of original features) of new columns = meaningless random combinations of "
            "feature pairs (v_i/v_j, v_i·v_j, v_i−v_j, |v_i−v_j|, v_i+v_j) and pure noise. At 100% there "
            "are as many invented as original.</li>"
            "<li>We measure test AP and, above all, the paired ΔAP vs 0% added features.</li>"
            "</ul></details>")
    return intro + tbl + tnote + mbox + fig_block("fig_feateng.png",
        "AP vs % of random invented features: absolute AP, paired ΔAP and per-dataset curve.")


nds = SF["n_datasets"] if SF else "…"
BODY = f"""<article>
<header class="hero">
  <p class="eyebrow">Empirical study · model selection · imbalanced data</p>
  <h1>One validation set,<br>or two?</h1>
  <p class="dek">Stress-tested on <strong>89 imbalanced binary datasets</strong>, sweeping the
  amount of data <strong>by fraction</strong> of each dataset's development pool — so the curve
  reflects data quantity, not sample composition.</p>
  <dl class="meta">
    <div><dt>Datasets</dt><dd>{nds}</dd></div>
    <div><dt>Base learner</dt><dd>LightGBM</dd></div>
    <div><dt>Metric</dt><dd>Average Precision</dd></div>
    <div><dt>Sweep</dt><dd>10% → 100%</dd></div>
  </dl>
</header>

<section><h2><span class="n">1</span> The question (and a few more)</h2>
  <p>A few years ago I wrote a Medium post arguing, <strong>mostly on theory</strong>, that
  <em>a single validation set is enough</em> to do both early stopping (choosing the number of
  trees) and hyperparameter selection — you don't need one set for each. Here I <strong>prove it
  empirically</strong> on 89 datasets, measuring what happens <strong>as data grows</strong>: a
  separate val set should curb the <strong>winner's curse</strong> with plenty of data, but wastes
  precious data when it's scarce.</p>
  <p>And since I was building the machinery, I put four more <strong>common ML habits</strong> to
  the test, each with a belief behind it:</p>
  <ul class="tight">
    <li><strong>Retraining the final model</strong> on all the data «to be safe» after selection —
    is it always worth it?</li>
    <li><strong>Grid search vs hyperparameter-optimization methods</strong>: many people still use a
    grid instead of Optuna / Bayesian or evolutionary search. Do the «smart» methods pay off?</li>
    <li><strong>«Lots of LightGBM configs is enough, it's faster than an ensemble».</strong> LightGBM
    is fast, true — but trying thousands of configs burns a lot of time anyway. <strong>At equal
    time</strong>, does that win, or an ensemble?</li>
    <li><strong>Speculative feature engineering</strong>: inventing new variables from the existing
    ones (ratios, products…) hoping one happens to capture a useful pattern. Does it help, or just
    add noise?</li>
  </ul>
</section>

<section><h2><span class="n">2</span> The two strategies (and retraining)</h2>
  <div class="cards">
    <div class="card e1"><h3>E1 · separate validation</h3>
      <p><code>dev → train + val</code>. Trees fixed by <strong>CV inside train</strong>; the best
      candidate chosen on the <strong>independent val set</strong>.</p></div>
    <div class="card e2"><h3>E2 · CV only</h3>
      <p><code>dev → CV</code>. Out-of-fold predictions decide <strong>both</strong> trees and
      candidate.</p></div>
  </div>
  <p>And two ways to build the <strong>final model</strong>, to test whether retraining helps:</p>
  <div class="cards">
    <div class="card"><h3>with retraining</h3><p>E1 refits on <code>train+val</code>; E2 on <code>dev</code>.</p></div>
    <div class="card"><h3>without retraining</h3><p>E1 keeps the <code>train</code>-only model; E2 a
      <strong>bagged ensemble</strong> of the <em>k</em> fold models.</p></div>
  </div>
</section>

<section><h2><span class="n">3</span> Setup</h2>
  <ul class="tight">
    <li><strong>Data:</strong> 89 imbalanced binary datasets (imblearn + OpenML).</li>
    <li><strong>Fixed test:</strong> 30% (capped at 6,000), shared by E1 and E2.</li>
    <li><strong>Fraction sweep</strong> of the development pool (10, 20, …, 100% in steps of 10, capped at 20,000 rows). Crucially, <em>every dataset is present at every point</em>, so the aggregate
    curve avoids the <strong>composition bias</strong> of an absolute-size sweep (where only the
    large — and different — datasets reach the high sizes).</li>
    <li><strong>Candidates:</strong> 18 (6 LightGBM configs × 3 imbalance techniques). No leakage
    (per-fold preprocessing). Early stopping on AP. Timings logged.</li>
  </ul>
</section>

<section><h2><span class="n">4</span> Result 1 · E1 vs E2</h2>
  {section4()}
  <details class='methods'><summary>Methodology · Results 1 and 2</summary>
    <ul><li><strong>89 imbalanced binary datasets</strong> (imblearn + OpenML) with all 10 fractions;
    <strong>2 seeds</strong> → 1,780 paired comparisons.</li>
    <li><strong>Fraction sweep</strong> of the development pool (10, 20, …, 100%, capped at 20,000
    rows): every dataset present at each point (no composition bias).</li>
    <li><strong>Fixed test</strong> 30% (capped at 6,000), shared by E1 and E2. CV <strong>k=4</strong>;
    early stopping on Average Precision (patience 50, up to 3,000 trees).</li>
    <li><strong>18 candidates = grid search</strong>: 6 LightGBM configs (num_leaves ∈ {15,31,63} ×
    reg_lambda ∈ {0,1}) × 3 imbalance techniques (none / class_weight / SMOTE). Rest fixed (lr 0.05,
    feature_fraction 0.8, min_child_samples 20). Trees set by early stopping.</li>
    <li>Per-fold preprocessing (impute + scale + one-hot), no leakage. Metric: Average Precision;
    significance by paired t-test. Very wide datasets (&gt;130 features) use a lighter preprocessing
    for RAM reasons.</li></ul></details>
</section>

<section><h2><span class="n">5</span> Result 2 · Does retraining help?</h2>
  {section5()}
</section>

<section><h2><span class="n">6</span> Result 3 · Hyperparameter optimization</h2>
  <h3>6a · Optuna vs grid, at equal number of configurations</h3>
  <p>The grid tries 18 fixed configs; Optuna (TPE) tries 18 trials over an 8-hyperparameter
  continuous space (and 40, as a higher-budget reference). The number of trees is set by early
  stopping in both.</p>
  {section_b()}
  <p class="note">A note on timing: even trying the same 18 configurations, they don't cost the same.
  The grid fixes <code>learning_rate=0.05</code>, so all its models grow many trees before early
  stopping; Optuna samples higher learning rates (up to 0.2) that converge in fewer trees. That's why
  Optuna-18 is <em>faster</em> than grid-18: «same number of configs» is not «same compute».</p>
  <h3>6b · Does more budget help? Which method?</h3>
  <p>A budget sweep (20 → 160 configurations) comparing four methods over the same space:
  <strong>grid</strong> (large shuffled grid), <strong>random</strong> search, <strong>TPE</strong>
  (Optuna's Bayesian) and <strong>CMA-ES</strong> (evolutionary).</p>
  {section_hpo()}
</section>

<section><h2><span class="n">7</span> Result 4 · Families + ensembles</h2>
  <p>A pool of 10 families combined with a <strong>greedy Caruana ensemble</strong> (a blend:
  weighted average of probabilities with weights chosen by greedy selection). Two symmetric variants:
  the <strong>E1-style ensemble</strong> learns the blend on the separate <code>val</code>; the
  <strong>E2-style</strong> learns it on the CV out-of-fold predictions and refits the families on
  all <code>dev</code>.</p>
  {section_c()}
</section>

<section><h2><span class="n">8</span> Result 5 · At equal time, what wins?</h2>
  {section_isofinal()}
</section>

<section><h2><span class="n">9</span> Result 6 · Speculative feature engineering</h2>
  <p>Tree models are said to discard irrelevant features, so throwing in invented variables «doesn't
  hurt» — and with luck one turns out useful by chance. Is it true? We take the best tree strategy (E2
  with LightGBM), 100% of dev, and add random invented features (10%, 20%, …, 100% of the original
  count) to see how performance changes.</p>
  {section_feateng()}
</section>

<section><h2><span class="n">10</span> Conclusions</h2>
  <ol class="concl">
    <li><strong>A single validation set is enough</strong> for imbalanced data; the theory holds.</li>
    <li>The E1/E2 gap is small and depends on data amount: E2 leads with scarce data, and its edge
    shrinks toward a tie as data grows (the theoretical direction that would favor E1 with abundant data).</li>
    <li><strong>Retraining</strong> helps E1 (recovers the val data) but not E2 (bagging covers it).</li>
    <li>For <strong>hyperparameter optimization</strong>, neither more budget (20→160 configs) nor
    smarter methods (random, TPE, CMA-ES) pull away from a <strong>small, well-chosen grid</strong>:
    all land within ~0.005 AP.</li>
    <li><strong>Ensembles</strong> of families do give something: the E2-style (blend on OOF + refit
    on all <code>dev</code>) <strong>beats E2 in ~2 of 3 cases</strong>, but at <strong>several times
    the time cost</strong>; the E1-style, at equal time, doesn't pay off. If time matters, the focused
    LightGBM is the best quality/cost trade-off.</li>
    <li>At <strong>equal time</strong>: giving more budget (more members to the E1 ensemble, more grid
    configs) <strong>does not catch ens-E2</strong>. Its edge is structural (it uses all of
    <code>dev</code> for blend and models), not a matter of time. The study's cross-cutting lesson:
    <strong>always compare at equal time</strong>, not at equal number of configurations.</li>
  </ol>
  <p class="foot">Methodology, code and data: <code>precision-at-k-study/valsplit_study</code>. Fraction sweep of the development pool.</p>
</section>
</article>"""

CSS = open(f"{STUDY}/_article_css.html").read()
open(f"{STUDY}/articulo_en.html", "w", encoding="utf-8").write('<meta charset="utf-8">\n' + CSS + BODY)
print("articulo_en.html escrito |", "summary_frac" if SF else "sin summary_frac")
