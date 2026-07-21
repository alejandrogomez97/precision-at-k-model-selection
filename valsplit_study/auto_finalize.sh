#!/bin/bash
# Orquestador nocturno autónomo: espera a que terminen hpo + ens_e2f, rellena
# CMA-ES, corre los análisis y reconstruye los dos artículos en local.
# Todo detached (setsid) y resumible. NO republica (eso lo hace Claude al volver).
set -u
cd /home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=../benchmark/env/bin/python
LOG=logs/auto_finalize.log
exec >> "$LOG" 2>&1
echo "===== AUTO-FINALIZE START $(date) ====="

alive () { ps -eo command | grep -E "$1" | grep -v 'grep\|bash -c' | wc -l; }

echo "-- esperando a hpo + ens_e2f --"
while [ "$(alive 'hpo_study|families_study')" -gt 0 ]; do sleep 60; done
echo "-- hpo + ens_e2f terminados $(date) --"

# --- relleno CMA-ES (proceso nuevo -> cmaes ya instalado; resumible) ---
DS="jm1 online_shoppers_intention Bank_marketing_data_set_UCI Pulsar-Dataset-HTRU2 letter satimage mammography rl"
echo "-- relleno cmaes $(date) --"
i=0
for d in $DS; do
  grp=$((i%3)); G[$grp]="${G[$grp]:-} $d"; i=$((i+1))
done
pids=()
for g in 0 1 2; do
  [ -z "${G[$g]:-}" ] && continue
  $PY hpo_study.py ${G[$g]} --seeds 0,1 --fracs 0.3,1.0 --methods cmaes --tag hpo \
      > logs/hpo_cmaes_$g.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "-- cmaes rellenado $(date) --"

# --- análisis + reconstrucción de artículos ---
echo "-- análisis $(date) --"
$PY analyze_hpo.py      || echo "analyze_hpo fallo"
$PY analyze_isotime.py  || echo "analyze_isotime fallo"
$PY analyze_bc.py both  || echo "analyze_bc fallo"
$PY analyze_frac.py     || echo "analyze_frac fallo"
$PY build_article_html.py    || echo "build ES fallo"
$PY build_article_html_en.py || echo "build EN fallo"
echo "===== AUTO-FINALIZE DONE $(date) ====="
touch AUTOFINAL_DONE
