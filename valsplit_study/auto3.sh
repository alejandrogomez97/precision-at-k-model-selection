#!/bin/bash
# Re-ejecuta las versiones @t* cogiendo la config que CRUZA t* (usa >= t*), con más
# presupuesto (grid MAXC=800, pool ens-E1 mayor). Luego re-analiza y reconstruye.
set -u
cd /home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=../benchmark/env/bin/python
exec >> logs/auto3.log 2>&1
echo "===== AUTO3 START $(date) ====="
alive(){ ps -eo command | grep -E "$1" | grep -v 'grep\|bash -c' | wc -l; }

rm -f results/isoE1__*.json results/isogrid__*.json
echo "-- borrados @t* antiguos; lanzo isoE1 (2w) + isogrid (3w) --"
DS="jm1 online_shoppers_intention Bank_marketing_data_set_UCI Pulsar-Dataset-HTRU2 letter satimage mammography rl"
# isoE1: 2 workers
E0=$(echo $DS|cut -d' ' -f1,3,5,7); E1=$(echo $DS|cut -d' ' -f2,4,6,8)
$PY -c "import isotime_ensE1 as I; I.DATASETS='$E0'.split(); I.main()" > logs/isoE1n_0.log 2>&1 &
$PY -c "import isotime_ensE1 as I; I.DATASETS='$E1'.split(); I.main()" > logs/isoE1n_1.log 2>&1 &
# isogrid: 3 workers
G0=$(echo $DS|cut -d' ' -f1,4,7); G1=$(echo $DS|cut -d' ' -f2,5,8); G2=$(echo $DS|cut -d' ' -f3,6)
$PY isotime_grid.py $G0 > logs/isogridn_0.log 2>&1 &
$PY isotime_grid.py $G1 > logs/isogridn_1.log 2>&1 &
$PY isotime_grid.py $G2 > logs/isogridn_2.log 2>&1 &
wait
echo "-- re-ejecución terminada $(date) --"
$PY analyze_isofinal.py || echo "isofinal fallo"
$PY build_article_html.py    || echo "build ES fallo"
$PY build_article_html_en.py || echo "build EN fallo"
echo "===== AUTO3 DONE $(date) ====="
touch AUTO3_DONE
