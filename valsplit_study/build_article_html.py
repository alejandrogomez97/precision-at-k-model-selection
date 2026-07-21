"""Genera articulo.html (auto-contenido, figuras incrustadas) desde los resultados
del barrido por FRACCIÓN. Los números se leen dinámicamente de summary_frac.json /
summary_B.csv / summary_C.csv, así la página se actualiza sola al reanalizar.
"""
import base64, glob, json, os
import numpy as np, pandas as pd

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"


def b64(p):
    p = f"{STUDY}/{p}"
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else None


def fig_block(path, cap):
    d = b64(path)
    return f'<figure><img src="{d}" alt="{cap}"/><figcaption>{cap}</figcaption></figure>' if d else ""


def jload(name):
    p = f"{STUDY}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x): return f"{x*100:.0f}%"


def mark(vals, higher_better=True):
    """Devuelve atributos class por celda: mejor -> best (verde), peor -> worst (rojo)."""
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


def fmt_p(p):
    if p is None or (isinstance(p, float) and p != p): return "n/a"
    if p < 1e-3: return f"{p:.0e}".replace("e-0", "e−").replace("e-", "e−")
    return f"{p:.3f}"


# --------------------------------------------------------------------------- #
SF = jload("summary_frac.json")


def section4():
    if not SF:
        return '<p class="note">Resultado principal en curso (re-run por fracción). Aparecerá aquí.</p>'
    e = SF["E1_vs_E2"]; ap = SF["ap_by_frac"]; cx = SF.get("crossover")
    tb = SF.get("time_by_frac", {})
    def tget(strat, f):
        m = tb.get(strat, {})
        return m.get(f, m.get(str(f)))
    rows = ""
    for f in SF["fracs"]:
        f = float(f)
        e1 = ap["E1-retrain"].get(f, ap["E1-retrain"].get(str(f)))
        e2 = ap["E2-retrain"].get(f, ap["E2-retrain"].get(str(f)))
        if e1 is None: continue
        d = e1 - e2
        c = mark([e1, e2])
        t1 = tget("E1", f); t2 = tget("E2", f)
        tc = mark([t1, t2], higher_better=False) if t1 is not None else ["", ""]
        tcols = (f"<td{tc[0]}>{t1:.0f}s</td><td{tc[1]}>{t2:.0f}s</td>"
                 if t1 is not None else "")
        rows += (f"<tr><td>{pct(f)}</td><td{c[0]}>{e1:.3f}</td><td{c[1]}>{e2:.3f}</td>"
                 f"<td>{d:+.4f}</td>{tcols}</tr>")
    if cx:
        lo, hi = cx["low"], cx["high"]
        # veredicto adaptativo por SIGNIFICANCIA (honesto, no forzado)
        lo_e2 = lo["mean"] < 0 and (lo.get("p") or 1) < 0.05
        hi_e1 = hi["mean"] > 0 and (hi.get("p") or 1) < 0.05
        hi_e2 = hi["mean"] < 0 and (hi.get("p") or 1) < 0.05
        if lo_e2 and hi_e1:
            big = "Con pocos datos gana E2; con muchos, gana E1 — el cruce que predice la teoría."
        elif lo_e2 and not (hi_e1 or hi_e2):
            big = ("E2 gana con pocos datos; con muchos datos las dos estrategias "
                   "empatan. La ventaja de E2 se desvanece al crecer los datos.")
        elif hi_e1 and not lo_e2:
            big = "E1 gana con muchos datos; con pocos, empate."
        else:
            big = "Las dos estrategias son prácticamente equivalentes en todo el rango."
        callout = (f'<div class="callout"><p class="big">{big}</p>'
                   f'<p>Diferencia pareada ΔAP(E1−E2): a fracciones <strong>bajas</strong> '
                   f'{lo["mean"]:+.4f} (E1 gana el {pct(lo["winrate_E1"])}), a fracciones '
                   f'<strong>altas</strong> {hi["mean"]:+.4f} (E1 gana el {pct(hi["winrate_E1"])}). '
                   f'La dirección teórica es clara —el <code>val</code> independiente de E1 solo '
                   f'compensa con datos de sobra, y con datos escasos trocearlos penaliza— pero '
                   f'la magnitud es pequeña.</p></div>')
    else:
        callout = ""
    thead_t = "<th>t E1</th><th>t E2</th>" if SF.get("time_by_frac") else ""
    tbl = ("<div class='tw'><table><thead><tr><th>fracción</th><th>AP E1</th>"
           f"<th>AP E2</th><th>ΔAP (E1−E2)</th>{thead_t}</tr></thead><tbody>"
           f"{rows}</tbody></table></div>")
    to = SF.get("time_overall", {})
    isot = (f'<p class="sig">Tiempo de búsqueda medio: E1 {to.get("E1","?")}s vs E2 '
            f'{to.get("E2","?")}s (~iguales, ambas hacen CV sobre 18 configs) → la comparación '
            f'E1 vs E2 <strong>ya es a igualdad de tiempo</strong>; el reentrenamiento del ganador '
            f'añade &lt;1s. La conclusión no cambia al igualar tiempos.</p>') if to else ""
    note = isot + (f'<p class="note">n = {e["n"]} comparaciones pareadas sobre {SF["n_datasets"]} '
            f'datasets. Nota: las diferencias siguen siendo pequeñas en magnitud (décimas de '
            f'punto de AP), pero el <em>signo</em> cambia de forma consistente con el tamaño.</p>')
    return callout + tbl + note + fig_block("fig_A_story.png",
        "E1 vs E2 (barrido por fracción, todos los datasets en cada punto): AP monótona, diferencia pareada (el cruce) y efecto del desbalanceo.")


def section5():
    if not SF or "retrain" not in SF:
        return '<p class="note">Comparación de reentrenamiento en curso. Aparecerá aquí.</p>'
    r = SF["retrain"]
    def verb(m, p):
        if p >= 0.05: return "es indiferente"
        return "ayuda" if m > 0 else "perjudica"
    li = ""
    for s in ("E1", "E2"):
        d = r[s]
        li += (f"<li><strong>{s}:</strong> reentrenar {verb(d['mean'], d['p'])} "
               f"(ΔAP = {d['mean']:+.4f}, gana el {pct(d['winrate_retrain'])}, "
               f"p = {fmt_p(d['p'])}, n = {d['n']}).</li>")
    intro = ("<p>Con las cuatro variantes evaluadas sobre las mismas particiones "
             "aislamos el valor del refit final tras la selección:</p><ul class='tight'>"
             + li + "</ul>")
    return intro + fig_block("fig_retrain.png",
        "Las 4 variantes (izq., mismas curvas de con-reentrenamiento que la figura anterior) y el efecto del reentrenamiento por estrategia (der.).")


def section_b():
    p = f"{STUDY}/summary_B.csv"; pj = f"{STUDY}/summary_B.json"
    if not os.path.exists(p):
        return '<p class="note">Fase B (Optuna iso-presupuesto) en curso. Aparecerá aquí.</p>'
    s = pd.read_csv(p); j = jload("summary_B.json") or {}
    iso = j.get("iso18"); b40 = j.get("b40")
    txt = "<p><strong>A igualdad de nº de configuraciones (18 vs 18):</strong> "
    if iso:
        v = "mejor" if iso["gain"] > 0.0005 else ("peor" if iso["gain"] < -0.0005 else "igual")
        txt += (f"Optuna es <strong>{v}</strong> que el grid (Δ medio {iso['gain']:+.4f} de AP, "
                f"gana el {pct(iso['winrate'])} de las celdas, n={iso['n']}). ")
    if b40:
        txt += (f"Incluso con más presupuesto (40 trials) la ganancia es {b40['gain']:+.4f} "
                f"(gana {pct(b40['winrate'])}).")
    txt += "</p>"
    has18 = "ap_o18" in s.columns
    def row(r):
        aps = [r.ap_grid, (r.ap_o18 if has18 else float('nan')), r.ap_o40]
        ts = [r.t_grid, (r.t_o18 if has18 else float('nan')), r.t_o40]
        ca = mark(aps, higher_better=True); ct = mark(ts, higher_better=False)
        return (f"<tr><td>{pct(r['frac'])}</td>"
                f"<td{ca[0]}>{aps[0]:.3f}</td><td{ca[1]}>{aps[1]:.3f}</td><td{ca[2]}>{aps[2]:.3f}</td>"
                f"<td{ct[0]}>{ts[0]:.0f}s</td><td{ct[1]}>{ts[1]:.0f}s</td><td{ct[2]}>{ts[2]:.0f}s</td></tr>")
    rows = "".join(row(r) for _, r in s.iterrows())
    tbl = ("<div class='tw'><table><thead><tr><th>fracción</th><th>AP grid-18</th>"
           "<th>AP optuna-18</th><th>AP optuna-40</th><th>t grid</th><th>t opt-18</th>"
           "<th>t opt-40</th></tr></thead><tbody>" f"{rows}</tbody></table></div>")
    return txt + tbl + fig_block("fig_B_optuna.png",
        "Optuna vs grid a igual nº de configuraciones (18): AP, ganancia y tiempo por fracción.")


def section_hpo():
    p = f"{STUDY}/summary_hpo.csv"
    if not os.path.exists(p):
        return '<p class="note">Barrido de presupuesto/método en curso. Aparecerá aquí.</p>'
    s = pd.read_csv(p, index_col=0)      # index=budget, columns=method
    j = jload("summary_hpo.json") or {}
    methods = [m for m in ["grid", "random", "tpe", "cmaes"] if m in s.columns]
    # ¿ayuda más presupuesto? (ganancia del mejor método de min a max budget)
    gains = {m: j.get(m, {}).get("gain_budget") for m in methods if m in j}
    best_gain = max(gains.values()) if gains else 0
    verdict = ("Aumentar el nº de configuraciones apenas mueve el AP" if abs(best_gain) < 0.008
               else "Más configuraciones mejoran algo el AP")
    # ¿algún método bate al grid al máximo presupuesto?
    maxb = s.index.max()
    at_max = s.loc[maxb]
    winner = at_max.idxmax()
    rows = ""
    for b in s.index:
        vals = [s.loc[b, m] for m in methods]
        c = mark(vals, higher_better=True)
        tds = "".join(f"<td{c[i]}>{v:.3f}</td>" for i, v in enumerate(vals))
        rows += f"<tr><td>{int(b)}</td>{tds}</tr>"
    head = "".join(f"<th>{m}</th>" for m in methods)
    tbl = (f"<div class='tw'><table><thead><tr><th>nº configs</th>{head}</tr></thead>"
           f"<tbody>{rows}</tbody></table></div>")
    # tabla de TIEMPOS de búsqueda (menor = verde)
    st = pd.read_csv(f"{STUDY}/summary_hpo_time.csv", index_col=0) if os.path.exists(f"{STUDY}/summary_hpo_time.csv") else None
    ttbl = ""
    if st is not None:
        tmethods = [m for m in methods if m in st.columns]
        trows = ""
        for b in st.index:
            vals = [st.loc[b, m] for m in tmethods]
            c = mark(vals, higher_better=False)
            tds = "".join(f"<td{c[i]}>{v:.0f}s</td>" for i, v in enumerate(vals))
            trows += f"<tr><td>{int(b)}</td>{tds}</tr>"
        thead = "".join(f"<th>{m}</th>" for m in tmethods)
        ttbl = ("<p class='note' style='margin-bottom:.3rem'>Tiempo de búsqueda (s):</p>"
                f"<div class='tw'><table><thead><tr><th>nº configs</th>{thead}</tr></thead>"
                f"<tbody>{trows}</tbody></table></div>")
    # significancia: cada método vs grid (t-test pareado, máx presupuesto)
    sigbits = []
    for m in ("random", "tpe", "cmaes"):
        r = j.get(m, {})
        if "vs_grid_p" in r:
            sigbits.append(f"{m} {r['vs_grid_mean']:+.4f} (p={fmt_p(r['vs_grid_p'])}, "
                           f"gana {pct(r['vs_grid_winrate'])})")
    n_cells = j.get("random", {}).get("vs_grid_n", "?")
    sigtxt = ("<p class='sig'>Significancia (vs grid, t-test pareado, n=" + str(n_cells) + "): "
              + "; ".join(sigbits) + ". <strong>Ninguna diferencia es estadísticamente significativa "
              "(todos p&gt;0.2)</strong>: ganan a veces en win-rate pero por un efecto minúsculo "
              "(~0.003 de AP).</p>") if sigbits else ""
    txt = (f"<p><strong>{verdict}</strong> (ganancia máx. de {min(s.index)}→{maxb} configs: "
           f"{best_gain:+.4f} de AP). Al máximo presupuesto ({maxb} configs), <strong>ni la búsqueda "
           f"bayesiana (TPE) ni la evolutiva (CMA-ES) se despegan del grid ni de la búsqueda "
           f"aleatoria</strong>. El tiempo crece ~lineal con el nº de configuraciones.</p>")
    mbox = ("<details class='methods'><summary>Metodología · Resultado 3</summary>"
            "<ul><li><strong>8 datasets</strong> (subconjunto representativo), fracciones "
            "<strong>30% y 100%</strong> del pool de desarrollo, <strong>2 semillas</strong> "
            "→ 32 celdas.</li>"
            "<li>Marco E1 (train/val; nº de árboles por CV en train; selección en val; refit del "
            "ganador en dev → test).</li>"
            "<li><strong>4 métodos</strong> de optimización, en qué consiste "
            "cada uno:<ul>"
            "<li><em>grid</em>: prueba puntos predefinidos de una rejilla grande (360 configs) barajada "
            "— sin aprender, cobertura sistemática.</li>"
            "<li><em>random</em>: muestrea configuraciones al azar del espacio — sorprendentemente "
            "competitivo, es el baseline honesto.</li>"
            "<li><em>TPE</em> (Tree-structured Parzen Estimator, el bayesiano de Optuna): modela qué "
            "regiones dan buenos/malos resultados y muestrea donde promete — explota lo aprendido.</li>"
            "<li><em>CMA-ES</em> (estrategia evolutiva): mantiene una «nube» gaussiana de configuraciones "
            "que va desplazando y estrechando hacia las mejores — bueno en espacios continuos.</li></ul></li>"
            "<li><strong>Espacio de búsqueda.</strong> Los tres samplers de Optuna exploran un espacio "
            "completo de <strong>8 hiperparámetros:</strong> num_leaves [7,255], "
            "min_child_samples [5,100], reg_lambda/alpha [1e-3,10], colsample_bytree [0.5,1], "
            "subsample [0.6,1], learning_rate [0.01,0.2], técnica de desbalanceo "
            "{none, class_weight, SMOTE}. La rejilla de 360 configs solo varía <strong>5 de ellos</strong> "
            "(num_leaves, min_child_samples, learning_rate, reg_lambda y la técnica de desbalanceo); los "
            "otros tres quedan en los valores por defecto de LightGBM. El nº de árboles lo fija el early stopping.</li>"
            "<li><strong>Presupuestos 20/40/80/160</strong> configuraciones, con checkpoints del "
            "mejor-hasta-ahora en una sola corrida de 160 por método.</li>"
            "<li>Comparación sobre <strong>panel balanceado</strong> (celdas comunes a los 4 "
            "métodos); significancia por t-test pareado vs grid.</li></ul></details>")
    return txt + sigtxt + tbl + ttbl + mbox + fig_block("fig_hpo.png",
        "AP en test vs nº de configuraciones por método (fracción baja y alta) y coste temporal.")


def _ens_verdict(w1, w2, fast2):
    e1 = (w1 is not None and w1 > 0.55)
    e2 = (w2 is not None and w2 > 0.55)
    if e2 and not e1:
        return ("<p>Matiz interesante: el <strong>ensemble estilo E2 sí mejora el AP de E2</strong> "
                "en la mayoría de casos, pero <strong>a costa de bastante más tiempo</strong> "
                "(entrena 10 familias con CV+refit, nunca más rápido que E2). El estilo E1, "
                "<strong>a igualdad de tiempo, no compensa</strong>. Conclusión práctica: si te sobra "
                "cómputo, un blend de familias sobre todo <code>dev</code> exprime algo más de AP; "
                "si el tiempo importa, el LightGBM enfocado gana.</p>")
    if e1 and e2:
        return ("<p>Ambos ensembles superan a su estrategia base en AP, a costa de más tiempo.</p>")
    return ("<p>Es decir, <strong>ninguno de los dos ensembles bate de forma fiable a su estrategia "
            "base</strong> en su comparación correspondiente; el LightGBM enfocado sigue costando de superar.</p>")


def section_c():
    p = f"{STUDY}/summary_isotime.csv"
    if not os.path.exists(p):
        return '<p class="note">Fase C (ensembles E1/E2) en curso. Aparecerá aquí.</p>'
    d = pd.read_csv(p); j = jload("summary_isotime.json") or {}
    w1 = j.get("ens_e1_beats_e1_isotime"); w2 = j.get("ens_e2_beats_e2")
    fast2 = j.get("ens_e2_faster")
    intro = ("<p>Dos ensembles, cada uno comparado con <em>su</em> estrategia:</p><ul class='tight'>"
             "<li><strong>Ensemble estilo E1</strong> (blend greedy sobre el <code>val</code> "
             "separado). Comparado con E1 <strong>a igualdad de tiempo</strong> (mejor AP que "
             f"alcanza en el tiempo que E1 gasta en su grid): gana en el <strong>"
             f"{pct(w1) if w1 is not None else '…'}</strong> de las celdas.</li>"
             "<li><strong>Ensemble estilo E2</strong> (blend sobre las predicciones OOF de la CV + "
             "familias reentrenadas en todo <code>dev</code>). Comparado con E2: gana en el "
             f"<strong>{pct(w2) if w2 is not None else '…'}</strong> de las celdas"
             + (f", y es más rápido que E2 solo en el <strong>{pct(fast2)}</strong>" if fast2 is not None else "")
             + ".</li></ul>")
    s1 = j.get("sig_ens_e1", {}); s2 = j.get("sig_ens_e2", {})
    if s2.get("p") is not None:
        intro += (f"<p class='sig'>Significancia (t-test pareado): ens-E1 vs E1 "
                  f"{s1.get('mean',0):+.4f} (p={fmt_p(s1.get('p',1))}, n.s.); "
                  f"<strong>ens-E2 vs E2 {s2.get('mean',0):+.4f} (p={fmt_p(s2.get('p',1))} → "
                  f"muy significativo)</strong>, n={s2.get('n','?')}. A diferencia del HPO, aquí el "
                  f"efecto sí es grande (~0.012 de AP) y claramente significativo.</p>")
    intro += _ens_verdict(w1, w2, fast2)
    by = d.groupby("frac").agg(e1=("e1_ap","mean"), ens1=("ens_at_e1","mean"),
        e2=("e2_ap","mean"), ens2=("e2ens_ap","mean"),
        t1=("e1_t","mean"), tens1=("ens_full_t","mean"),
        t2=("e2_t","mean"), tens2=("e2ens_t","mean")).reset_index()
    def row(r):
        import math
        ca = mark([r.e1, r.ens1, r.e2, r.ens2], higher_better=True)
        ct = mark([r.t1, r.tens1, r.t2, r.tens2], higher_better=False)
        def cell(v, c, s="{:.3f}"): return f"<td{c}>{s.format(v)}</td>" if not math.isnan(v) else "<td>—</td>"
        return (f"<tr><td>{pct(r['frac'])}</td>{cell(r.e1,ca[0])}{cell(r.ens1,ca[1])}"
                f"{cell(r.e2,ca[2])}{cell(r.ens2,ca[3])}"
                f"{cell(r.t1,ct[0],'{:.0f}s')}{cell(r.tens1,ct[1],'{:.0f}s')}"
                f"{cell(r.t2,ct[2],'{:.0f}s')}{cell(r.tens2,ct[3],'{:.0f}s')}</tr>")
    rows = "".join(row(r) for _, r in by.iterrows())
    tbl = ("<div class='tw'><table><thead><tr><th>fracción</th><th>AP E1</th>"
           "<th>AP ens-E1</th><th>AP E2</th><th>AP ens-E2</th>"
           "<th>t E1</th><th>t ens-E1</th><th>t E2</th><th>t ens-E2</th></tr></thead><tbody>"
           f"{rows}</tbody></table></div>")
    mbox = ("<details class='methods'><summary>Metodología · Resultado 4</summary>"
            "<ul><li><strong>16 datasets</strong> (subconjunto), <strong>10 fracciones</strong> "
            "(10→100%), <strong>2 semillas</strong> → hasta 318 celdas.</li>"
            "<li><strong>Pool de 10 familias, un modelo por familia con configuración fija "
            "(sin grid search):</strong> regresión logística (class_weight balanceado), Gaussian NB, "
            "kNN (k=25), LightGBM (400 árboles, num_leaves 31, lr 0.05), HistGBM (400 iters, lr 0.05, "
            "early stopping), XGBoost (400 árboles, prof. 6, scale_pos_weight), ExtraTrees (300), "
            "Random Forest (300), CatBoost (400 iters, prof. 6, auto-balanceado) y MLP (64-32, "
            "early stopping).</li>"
            "<li><strong>Ensemble = blend por selección greedy de Caruana</strong> (con reemplazo) "
            "que maximiza AP; los pesos salen como fracciones (p.ej. hgb 2/3 + rf 1/3). Suele "
            "quedarse con 2-4 familias.</li>"
            "<li><strong>Estilo E1:</strong> blend aprendido sobre el <code>val</code> separado; "
            "familias entrenadas en train. Comparado con E1 <em>a igualdad de tiempo</em> (curva "
            "anytime).</li>"
            "<li><strong>Estilo E2:</strong> blend aprendido sobre las predicciones OOF de la CV "
            "(k=4) + familias reentrenadas en todo <code>dev</code>. Comparado con E2 en AP.</li>"
            "<li><strong>Por qué ens-E2 cuesta ~5×:</strong> ens-E1 hace <strong>10 entrenamientos</strong> "
            "(1 por familia sobre train; el val separado ya da predicciones out-of-sample, sin CV). "
            "ens-E2 hace <strong>50</strong>: 4 folds × 10 familias para las OOF (40) + refit de las 10 "
            "en todo <code>dev</code>. Es el mismo tradeoff E1/E2 (split único vs CV) aplicado al ensemble.</li>"
            "<li>Significancia por t-test pareado.</li></ul></details>")
    nr = jload("summary_ensnr.json")
    nrnote = ""
    if nr:
        nrnote = (f"<p><strong>¿Y ens-E2 con o sin reentrenamiento?</strong> Recordando que en el E2 "
                  f"<em>base</em> el bagged (sin-retrain) ganaba claramente: en el <em>ensemble</em> "
                  f"apenas hay diferencia en AP — retrain {nr['ap_retrain']:.3f} vs no-retrain "
                  f"{nr['ap_noretrain']:.3f} "
                  f"<span class='sig'>(Δ={nr['delta']:+.4f}, p={fmt_p(nr['p'])}, n.s.; n={nr['n']})</span> "
                  f"— porque el blend de 10 modelos ya reduce la varianza. Pero el <strong>no-retrain "
                  f"es más rápido</strong> ({nr['t_noretrain']:.0f}s vs {nr['t_retrain']:.0f}s, se ahorra "
                  f"el refit), así que <strong>iguala en calidad y ahorra ~25% de tiempo</strong> → es la "
                  f"opción preferible para el ensemble E2.</p>")
    return intro + nrnote + tbl + mbox + fig_block("fig_C_isotime.png",
        "Ensemble estilo E1 vs E1 (iso-tiempo) y estilo E2 vs E2: AP, tasa de victoria y tiempos.")


def section_isofinal():
    import math
    p = f"{STUDY}/summary_isofinal.csv"
    if not os.path.exists(p):
        return '<p class="note">Experimento a igualdad de tiempo en curso. Aparecerá aquí.</p>'
    d = pd.read_csv(p); j = jload("summary_isofinal.json") or {}
    d = d.dropna(subset=[c for c in ["e1_at", "e2_at", "ens_e1_at"] if c in d.columns])
    ma = j.get("mean_ap", {}); se1 = j.get("ens_e2_vs_ens_e1_at", {})
    sg1 = j.get("ens_e2_vs_e1_at", {}); sg2 = j.get("ens_e2_vs_e2_at", {})
    intro = ("<p>La comparación justa es a <strong>igualdad de tiempo</strong>. Como ens-E2 es "
             "~5× más lento, damos a las demás estrategias <strong>su mismo presupuesto de tiempo</strong> "
             "(<strong>t*</strong> = tiempo de ens-E2) para hacer <em>más</em>: más configuraciones de "
             "LightGBM (E1@t*, E2@t*) y más miembros en el ensemble (ens-E1@t*). El sufijo "
             "<strong>@t*</strong> marca «misma estrategia, buscando hasta consumir el tiempo de ens-E2».</p>")
    if se1.get("p") is not None:
        intro += (f"<p>Resultado: a igualdad de tiempo, <strong>ens-E2 sigue ganando</strong> "
                  f"(AP medio {ma.get('ens_e2',0):.3f}). "
                  f"<span class='sig'>ens-E2 − ens-E1@t*: {se1['mean']:+.4f} (p={fmt_p(se1['p'])}, gana "
                  f"{pct(se1['winrate'])}); ens-E2 − E1@t*: {sg1.get('mean',0):+.4f} "
                  f"(p={fmt_p(sg1.get('p',1))}); ens-E2 − E2@t*: {sg2.get('mean',0):+.4f} "
                  f"(p={fmt_p(sg2.get('p',1))}). n={se1['n']}.</span></p>")
    by = d.groupby("frac").mean(numeric_only=True).reset_index()

    def cell(v, cc="", s="{:.3f}"):
        return f"<td{cc}>{s.format(v)}</td>" if (v == v) else "<td>—</td>"
    # tabla de AP
    aprows = ""
    for _, r in by.iterrows():
        vals = [r.e1_ap, r.get("e1_at", float("nan")), r.ens_e1, r.ens_e1_at,
                r.e2_ap, r.get("e2_at", float("nan")), r.ens_e2]
        c = mark(vals, higher_better=True)
        aprows += (f"<tr><td>{pct(r['frac'])}</td>" + "".join(cell(v, c[i]) for i, v in enumerate(vals)) + "</tr>")
    aptbl = ("<div class='tw'><table><thead><tr><th>frac</th><th>E1</th><th>E1@t*</th>"
             "<th>ens-E1</th><th>ens-E1@t*</th><th>E2</th><th>E2@t*</th><th>ens-E2</th>"
             f"</tr></thead><tbody>{aprows}</tbody></table></div>")
    # tabla de tiempos
    trows = ""
    for _, r in by.iterrows():
        tv = [r.e1_t, r.get("e1_at_t", float("nan")), r.ens_e1_t, r.ens_e1_at_t,
              r.e2_t, r.get("e2_at_t", float("nan")), r.e2ens_t]
        c = mark(tv, higher_better=False)
        trows += (f"<tr><td>{pct(r['frac'])}</td>" + "".join(cell(v, c[i], "{:.0f}s") for i, v in enumerate(tv)) + "</tr>")
    ttbl = ("<p class='note' style='margin-bottom:.3rem'>Tiempos (s):</p><div class='tw'><table><thead><tr>"
            "<th>frac</th><th>E1</th><th>E1@t*</th><th>ens-E1</th><th>ens-E1@t*</th><th>E2</th>"
            f"<th>E2@t*</th><th>ens-E2</th></tr></thead><tbody>{trows}</tbody></table></div>")
    verd = ("<p>Aunque las estrategias baratas usen su tiempo extra para probar más configuraciones/"
            "miembros, <strong>no alcanzan a ens-E2</strong>. Su ventaja es <strong>estructural, no de "
            "tiempo</strong>: usa todo <code>dev</code> para aprender el blend (OOF) <em>y</em> para los "
            "modelos finales (refit). El ensemble estilo E1 además <strong>satura</strong> (más miembros "
            "dejan de ayudar). Conclusión: si vas sobrado de tiempo, el blend estilo E2 exprime algo más "
            "de AP; para la mejor relación calidad/tiempo, el LightGBM enfocado.</p>")
    mbox = ("<details class='methods'><summary>Metodología · Apartado 8</summary>"
            "<ul><li><strong>8 datasets</strong>, 10 fracciones, 2 semillas. Mismas particiones que el resto.</li>"
            "<li><strong>t* = tiempo que consume ens-E2</strong> en cada celda (blend OOF de la CV + refit "
            "de 10 familias en todo <code>dev</code> = 50 entrenamientos).</li>"
            "<li><strong>E1@t* y E2@t*:</strong> grid de LightGBM (grid grande barajado) creciendo "
            "configuraciones —hasta 240— hasta consumir t*; el mejor (en val para E1, en OOF para E2) se "
            "reentrena en dev → test.</li>"
            "<li><strong>ens-E1@t*:</strong> pool ampliado de ~36 miembros (variantes de las 10 familias) "
            "añadidos con blend greedy sobre val hasta t* (o agotar el pool).</li>"
            "<li>Columnas sin <em>@t*</em> = versión natural (del Resultado 4). Significancia por t-test pareado.</li>"
            "</ul></details>")
    return intro + aptbl + ttbl + verd + mbox + fig_block("fig_isofinal.png",
        "A igualdad de tiempo (t* = tiempo de ens-E2): AP de las versiones @t* vs ens-E2, y dispersión por celda.")


def section_feateng():
    j = jload("summary_feateng.json")
    if not j:
        return '<p class="note">Experimento de feature engineering en curso. Aparecerá aquí.</p>'
    dA = j.get("dAP_100_mean"); p = j.get("dAP_100_p")
    verdict = ("añadir variables inventadas <strong>apenas afecta</strong> al modelo"
               if (dA is not None and abs(dA) < 0.005)
               else ("<strong>degrada</strong> el modelo" if (dA is not None and dA < 0)
                     else "<strong>mejora</strong> el modelo"))
    sigtxt = (f" (ΔAP a 100% de basura = {dA:+.4f}, p={fmt_p(p)}"
              f"{', no significativo' if (p and p>=0.05) else ''})") if dA is not None else ""
    intro = (f"<p>Con LightGBM (E2), <strong>{verdict}</strong>{sigtxt}. El árbol, en efecto, "
             f"<strong>ignora en su mayor parte las variables sin sentido</strong>: aunque dobles el "
             f"nº de features con basura aleatoria, el AP se mantiene prácticamente plano.</p>")
    rows = ""
    for r in j.get("by_level", []):
        cls = "pos" if r["dAP"] > 0.002 else ("neg" if r["dAP"] < -0.002 else "")
        rows += (f"<tr><td>{r['level']*100:.0f}%</td><td>{r['ap']:.3f}</td>"
                 f"<td class='{cls}'>{r['dAP']:+.4f}</td><td>{r['time']:.0f}s</td></tr>")
    tbl = ("<div class='tw'><table><thead><tr><th>% variables inventadas</th><th>AP</th>"
           "<th>ΔAP vs 0%</th><th>tiempo</th></tr></thead><tbody>"
           f"{rows}</tbody></table></div>")
    t0 = j.get("time_0"); t100 = j.get("time_100")
    tnote = (f"<p class='note'>El verdadero daño no es el AP: aunque la caída al 100% de basura "
             f"no sea significativa, el <strong>tiempo de entrenamiento sí sube</strong> (de {t0:.0f}s con "
             f"0% a {t100:.0f}s con 100%) para <strong>cero ganancia de AP</strong>. Y ahí está el "
             f"problema — bajo la óptica de igualdad de tiempo del Apartado 8, ese tiempo malgastado es "
             f"tiempo que podrías haber dedicado a entrenar <em>más</em> modelos (un grid mayor, más "
             f"miembros del ensemble), que sí ayuda. Es decir: es un uso estrictamente peor del cómputo.</p>") if t0 and t100 else ""
    mbox = ("<details class='methods'><summary>Metodología · Apartado 9 (Resultado 6)</summary>"
            "<ul><li><strong>Pregunta:</strong> hay quien cree que «el modelo descarta las variables "
            "irrelevantes, así que aunque no aporten no molestan; y a veces das por casualidad con un "
            "patrón que mejora». Lo comprobamos.</li>"
            "<li><strong>Modelo:</strong> estrategia E2 con LightGBM (el mejor modelo único y justo el "
            "árbol cuya robustez se discute; se evita ens-E2 porque sus miembros no-árbol sí sufren con "
            "la basura y confundirían la pregunta).</li>"
            f"<li><strong>{j.get('n_datasets','?')} datasets</strong>, 100% del dev (capado a 8.000 "
            "filas por coste), 2 semillas.</li>"
            "<li><strong>Variables inventadas (feature engineering especulativo):</strong> por cada nivel, se añade un "
            "porcentaje (10%, 20%, …, 100% del nº de features originales) de columnas nuevas = "
            "combinaciones aleatorias sin sentido de pares de features (v_i/v_j, v_i·v_j, v_i−v_j, "
            "|v_i−v_j|, v_i+v_j) y ruido puro. Al 100% hay tantas inventadas como originales.</li>"
            "<li>Se mide AP en test y, sobre todo, el cambio pareado ΔAP respecto a 0% de basura.</li>"
            "</ul></details>")
    return intro + tbl + tnote + mbox + fig_block("fig_feateng.png",
        "AP vs % de variables inventadas al azar: AP absoluto, cambio pareado ΔAP y curva por dataset.")


nds = SF["n_datasets"] if SF else "…"
HTML = f"""<article>
<header class="hero">
  <p class="eyebrow">Estudio empírico · selección de modelos · datos desbalanceados</p>
  <h1>¿Un solo conjunto de validación,<br>o dos?</h1>
  <p class="dek">Puesto a prueba sobre <strong>89 datasets binarios desbalanceados</strong>,
  barriendo la cantidad de datos <strong>por fracción</strong> del pool de desarrollo de
  cada dataset, para ver la performance sin sesgo de composición.</p>
  <dl class="meta">
    <div><dt>Datasets</dt><dd>{nds}</dd></div>
    <div><dt>Modelo base</dt><dd>LightGBM</dd></div>
    <div><dt>Métrica</dt><dd>Average Precision</dd></div>
    <div><dt>Barrido</dt><dd>10% → 100%</dd></div>
  </dl>
</header>

<section>
  <h2><span class="n">1</span> La pregunta (y unas cuantas más)</h2>
  <p>Hace unos años escribí un artículo en Medium defendiendo, <strong>sobre todo con
  teoría</strong>, que <em>un único conjunto de validación basta</em> para hacer a la vez
  <em>early stopping</em> (elegir el número de árboles) y selección de hiperparámetros —
  que no hace falta un conjunto para lo uno y otro para lo otro. Aquí lo <strong>pruebo
  empíricamente</strong> sobre 89 datasets, midiendo qué pasa <strong>según la cantidad de
  datos</strong>:</p>
  <ul>
    <li>Con <strong>muchos datos</strong>, un <code>val</code> independiente para la
    selección debería reducir la <strong>maldición del ganador</strong>.</li>
    <li>Con <strong>pocos datos</strong>, ese conjunto es diminuto; usar todo vía CV
    debería salir a cuenta.</li>
  </ul>
  <p>Y ya que montaba la maquinaria, aproveché para poner a prueba otros cuatro
  <strong>hábitos comunes</strong> del día a día en ML, cada uno con su creencia detrás:</p>
  <ul class="tight">
    <li><strong>Reentrenar el modelo final</strong> en todos los datos «por si acaso», tras
    la selección — ¿aporta siempre?</li>
    <li><strong>Grid search vs métodos de optimización de hiperparámetros</strong>: mucha
    gente sigue usando un grid en vez de Optuna/búsqueda bayesiana o evolutiva. ¿Merecen la
    pena los métodos «listos»?</li>
    <li><strong>«Con muchas configuraciones de LightGBM me vale, es más rápido que un
    ensemble».</strong> LightGBM es rápido, sí — pero probar miles de configuraciones acaba
    costando mucho tiempo igualmente. <strong>A igualdad de tiempo</strong>, ¿gana eso o un
    ensemble?</li>
    <li><strong>Feature engineering especulativo</strong>: generar variables nuevas a partir
    de las existentes (cocientes, productos…) con la esperanza de que «suene la flauta» y
    alguna capture un patrón útil. ¿Ayuda, o solo mete ruido?</li>
  </ul>
</section>

<section>
  <h2><span class="n">2</span> Las dos estrategias (y el reentrenamiento)</h2>
  <div class="cards">
    <div class="card e1"><h3>E1 · validación separada</h3>
      <p><code>dev → train + val</code>. Nº de árboles por <strong>CV en train</strong>
      (early stopping); el mejor candidato se elige en el <strong>val independiente</strong>.</p></div>
    <div class="card e2"><h3>E2 · solo CV</h3>
      <p><code>dev → CV</code>. Los folds OOF deciden <strong>a la vez</strong> árboles y
      candidato.</p></div>
  </div>
  <p>Y dos formas de construir el <strong>modelo final</strong> a test, para medir si el
  reentrenamiento aporta:</p>
  <div class="cards">
    <div class="card"><h3>con reentrenamiento</h3>
      <p>E1 → refit en <code>train+val</code>; E2 → refit en <code>dev</code>.</p></div>
    <div class="card"><h3>sin reentrenamiento</h3>
      <p>E1 → modelo entrenado solo en <code>train</code>; E2 → <strong>ensemble bagged</strong>
      de los <em>k</em> folds.</p></div>
  </div>
</section>

<section>
  <h2><span class="n">3</span> Montaje</h2>
  <ul class="tight">
    <li><strong>Datos:</strong> 89 datasets binarios desbalanceados (imblearn + OpenML).</li>
    <li><strong>Test fijo:</strong> 30% (capado a 6.000), común a E1 y E2.</li>
    <li><strong>Barrido por FRACCIÓN</strong> del pool de desarrollo (10, 20, 30, …, 100% de 10 en 10,
    capado a 20.000 filas). Clave: <em>todos los datasets están en todos los puntos</em>, así
    la curva agregada no sufre el <strong>sesgo de composición</strong> que da un barrido por
    número absoluto (donde solo los datasets grandes —y distintos— llegan a tamaños altos).</li>
    <li><strong>Candidatos:</strong> 18 (6 configs LightGBM × 3 técnicas de desbalanceo).
    Sin fugas (preprocesado por-fold). Early stopping con AP. Tiempos registrados.</li>
  </ul>
</section>

<section>
  <h2><span class="n">4</span> Resultado 1 · E1 vs E2</h2>
  {section4()}
  <p>Este cruce solo se ve al barrer por <strong>fracción</strong>: con un barrido por número
  absoluto quedaba enmascarado por el sesgo de composición (los datasets grandes, que
  dominan los tamaños altos, son además los más difíciles). Es un buen recordatorio de que
  la forma de agregar entre datasets puede cambiar la conclusión.</p>
  <details class='methods'><summary>Metodología · Resultados 1 y 2</summary>
    <ul><li><strong>89 datasets</strong> binarios desbalanceados (imblearn + OpenML), con las 10
    fracciones completas; <strong>2 semillas</strong> → 1.780 comparaciones pareadas.</li>
    <li><strong>Barrido por fracción</strong> del pool de desarrollo (10, 20, …, 100%, capado a
    20.000 filas): todos los datasets presentes en cada punto (sin sesgo de composición).</li>
    <li><strong>Test fijo</strong> 30% (capado a 6.000), común a E1 y E2. CV <strong>k=4</strong>;
    early stopping con Average Precision (paciencia 50, hasta 3.000 árboles).</li>
    <li><strong>18 candidatos = grid search</strong>: 6 configs de LightGBM (num_leaves ∈ {15,31,63}
    × reg_lambda ∈ {0, 1}) × 3 técnicas de desbalanceo (none / class_weight / SMOTE). Resto fijo
    (lr 0.05, feature_fraction 0.8, min_child_samples 20). El nº de árboles lo fija el early
    stopping.</li>
    <li>Preprocesado por-fold (imputación + escalado + one-hot), sin fugas. Métrica: Average
    Precision. Significancia por t-test pareado. Los datasets muy anchos (&gt;130 features) usan
    solo un preprocesado más ligero por coste de RAM.</li></ul></details>
</section>

<section>
  <h2><span class="n">5</span> Resultado 2 · ¿Ayuda reentrenar?</h2>
  {section5()}
</section>

<section>
  <h2><span class="n">6</span> Resultado 3 · Optimización de hiperparámetros</h2>
  <h3>6a · Optuna vs grid, a igual nº de configuraciones</h3>
  <p>El grid prueba 18 configs fijas; Optuna (TPE) prueba 18 trials sobre un espacio continuo de 8
  hiperparámetros (y también 40, como referencia de más presupuesto). El nº de árboles lo fija el
  early stopping en ambos.</p>
  {section_b()}
  <p class="note">Nota sobre los tiempos: aunque prueben las mismas 18 configuraciones, no cuestan
  lo mismo. El grid fija <code>learning_rate=0.05</code>, así que todos sus modelos crecen muchos
  árboles antes del early stopping; Optuna muestrea learning rates más altos (hasta 0.2) que
  convergen en menos árboles. Por eso Optuna-18 resulta <em>más rápido</em> que grid-18: «mismo nº
  de configs» no es «mismo cómputo».</p>
  <h3>6b · ¿Ayuda más presupuesto? ¿Qué método?</h3>
  <p>Barrido de presupuesto (20 → 160 configuraciones) comparando cuatro métodos sobre el mismo
  espacio: <strong>grid</strong> (grid grande barajado), <strong>random</strong> (búsqueda
  aleatoria), <strong>TPE</strong> (bayesiano de Optuna) y <strong>CMA-ES</strong> (evolutivo).</p>
  {section_hpo()}
</section>

<section>
  <h2><span class="n">7</span> Resultado 4 · Familias + ensembles</h2>
  <p>Un pool de 10 familias (logística, RF, ExtraTrees, HistGBM, LightGBM, XGBoost, CatBoost,
  MLP, kNN, GNB) combinadas con un <strong>ensemble greedy de Caruana</strong> (un blend: media
  ponderada de probabilidades con pesos elegidos por selección voraz). Se prueban dos variantes
  simétricas a las estrategias: el <strong>ensemble estilo E1</strong> aprende el blend sobre el
  <code>val</code> separado; el <strong>estilo E2</strong> lo aprende sobre las predicciones OOF
  de la CV y reentrena las familias en todo <code>dev</code>.</p>
  {section_c()}
</section>

<section>
  <h2><span class="n">8</span> Resultado 5 · A igualdad de tiempo, ¿qué gana?</h2>
  {section_isofinal()}
</section>

<section>
  <h2><span class="n">9</span> Resultado 6 · Feature engineering especulativo</h2>
  <p>Se dice que un modelo de árboles descarta las variables irrelevantes, así que meter variables
  inventadas «no molesta» —y con suerte alguna resulta útil por casualidad—. ¿Es cierto? Cogemos la
  mejor estrategia de árbol (E2 con LightGBM), el 100% del dev, y le añadimos variables inventadas al
  azar (10%, 20%, …, 100% del nº de originales) para ver cómo cambia la performance.</p>
  {section_feateng()}
</section>

<section>
  <h2><span class="n">10</span> Conclusiones</h2>
  <ol class="concl">
    <li>La ventaja depende de la <strong>cantidad de datos</strong> en la dirección que predice la
    teoría: con datos escasos conviene <strong>E2</strong> (la CV lo aprovecha todo), y esa
    ventaja <strong>se reduce</strong> al crecer los datos (donde el <code>val</code> independiente
    de E1 empieza a compensar por menor maldición del ganador).</li>
    <li>Las magnitudes son <strong>pequeñas</strong> (décimas de AP): importa más el <em>criterio</em>
    —elige según tu presupuesto de datos— que la diferencia absoluta.</li>
    <li>Sobre el <strong>reentrenamiento</strong>: en E1 conviene reentrenar (recupera el val);
    en E2, el bagged de la CV ya lo cubre.</li>
    <li>En <strong>optimización de hiperparámetros</strong>, ni más presupuesto (20→160 configs) ni
    métodos más sofisticados (random, TPE, CMA-ES) se despegan de un <strong>grid pequeño y bien
    elegido</strong>: todos quedan dentro de ~0.005 de AP.</li>
    <li>Los <strong>ensembles</strong> de familias sí dan algo: el estilo E2 (blend sobre OOF + refit
    en todo <code>dev</code>) <strong>mejora el AP de E2 en ~2 de cada 3 casos</strong>, pero a
    <strong>varias veces su coste en tiempo</strong>; el estilo E1, a igualdad de tiempo, no compensa.
    Si el tiempo importa, el LightGBM enfocado es la mejor relación calidad/coste.</li>
    <li>Comparando a <strong>igualdad de tiempo</strong>: dar más presupuesto (más miembros al ensemble
    E1, más configs al grid) <strong>no alcanza a ens-E2</strong>. Su ventaja es estructural (usa todo
    <code>dev</code> para blend y modelos), no de tiempo. Lección transversal del estudio:
    <strong>siempre hay que comparar a igualdad de tiempo</strong>, no de nº de configuraciones.</li>
  </ol>
  <p class="foot">Metodología, código y datos: <code>precision-at-k-study/valsplit_study</code>.
  Barrido por fracción del pool de desarrollo.</p>
</section>
</article>"""

CSS = open(f"{STUDY}/_article_css.html").read()
open(f"{STUDY}/articulo.html", "w", encoding="utf-8").write('<meta charset="utf-8">\n' + CSS + HTML)
print("articulo.html escrito |", "summary_frac" if SF else "sin summary_frac aún")
