#!/usr/bin/env bash
# Reproduce the whole study from scratch (fixed seeds). Datasets download on first run.
set -e
pip install -r requirements.txt

# main study (LightGBM, K-sweep over 93 datasets)
python experiment_real3.py        # -> results_real3.json
python meta_real3.py              # -> figT0/figT2/figT5, meta3_summary.json, Wilcoxon tests

# figures for the training-loss section
python fig_step.py                # -> fig1_step_function.png
python fig_gradient.py            # -> fig_gradient_example.png
python fig_apriori.py             # -> figT6_apriori.png

# robustness across model families (Random Forest + Logistic Regression)
python experiment_families.py     # -> results_families.json
python meta_families.py           # -> figF_families.png, families_summary.json
python fig_families_by_count.py   # -> figG_families_by_count.png (regret vs count, per family)

echo "Done. Figures: fig1_step_function.png, fig_gradient_example.png, figT0/figT2/figT5/figT6, figF_families.png, figG_families_by_count.png"
