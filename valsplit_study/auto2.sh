#!/bin/bash
# Encadena: espera isoE1 -> lanza isogrid (E1@t*, E2@t*) -> espera isogrid+feateng
# -> corre todos los análisis y reconstruye ambos artículos. Detached, resumible.
set -u
cd /home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=../benchmark/env/bin/python
exec >> logs/auto2.log 2>&1
echo "===== AUTO2 START $(date) ====="
alive(){ ps -eo command | grep -E "$1" | grep -v 'grep\|bash -c' | wc -l; }

echo "-- esperando isoE1 (ens-E1@t*) --"
while [ "$(alive 'isotime_ensE1')" -gt 0 ]; do sleep 60; done
echo "-- isoE1 hecho $(date); lanzo isogrid --"
DS="jm1 online_shoppers_intention Bank_marketing_data_set_UCI Pulsar-Dataset-HTRU2 letter satimage mammography rl"
G0=$(echo $DS|cut -d' ' -f1,4,7); G1=$(echo $DS|cut -d' ' -f2,5,8); G2=$(echo $DS|cut -d' ' -f3,6)
$PY isotime_grid.py $G0 > logs/isogrid_0.log 2>&1 &
p0=$!
$PY isotime_grid.py $G1 > logs/isogrid_1.log 2>&1 &
p1=$!
$PY isotime_grid.py $G2 > logs/isogrid_2.log 2>&1 &
p2=$!
wait $p0 $p1 $p2
echo "-- isogrid hecho $(date) --"

echo "-- esperando feateng --"
while [ "$(alive 'feature_eng')" -gt 0 ]; do sleep 60; done
echo "-- feateng hecho $(date); análisis + rebuild --"
$PY analyze_frac.py       || echo "frac fallo"
$PY analyze_bc.py both    || echo "bc fallo"
$PY analyze_isotime.py    || echo "isotime fallo"
$PY analyze_isofinal.py   || echo "isofinal fallo"
$PY analyze_hpo.py        || echo "hpo fallo"
$PY analyze_feateng.py    || echo "feateng fallo"
$PY build_article_html.py    || echo "build ES fallo"
$PY build_article_html_en.py || echo "build EN fallo"
echo "===== AUTO2 DONE $(date) ====="
touch AUTO2_DONE
