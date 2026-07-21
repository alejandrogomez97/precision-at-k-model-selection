#!/bin/bash
# Orquestador nocturno: Fase A (E1/E2 grid, breadth-first por semilla) ->
# Fase B (Optuna, subconjunto) -> Fase C (familias+ensembles, subconjunto).
# Todo resumible: si algo ya está calculado, se salta.
set -u
cd /home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=../benchmark/env/bin/python
W=6
mkdir -p logs

run_parallel () {  # $1 = fichero con comandos (uno por línea)
  local pids=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    eval "$line &"
    pids+=($!)
  done < "$1"
  for p in "${pids[@]}"; do wait "$p"; done
}

echo "===== FASE A: E1 vs E2 (grid) — $(date) ====="
for SEED in 0 1 2; do
  echo "--- Fase A pasada seed=$SEED $(date) ---"
  $PY launch.py grid $W --seeds $SEED > /tmp/cmds_A_s$SEED.sh
  # renombrar logs por semilla para no pisarlos
  sed -i "s#logs/grid_w\([0-9]*\).log#logs/gridA_w\1_s$SEED.log#g" /tmp/cmds_A_s$SEED.sh
  run_parallel /tmp/cmds_A_s$SEED.sh
done
echo "===== FASE A COMPLETA — $(date) ====="
$PY analyze.py grid > logs/analyze_A.log 2>&1 || true

# Subconjunto representativo (variedad de tamaño y desbalanceo) para B y C
SUBSET="jm1 online_shoppers_intention Bank_marketing_data_set_UCI pendigits \
Pulsar-Dataset-HTRU2 Give-Me-Some-Credit-Sampled-Preprocessed letter satimage \
coil_2000 mammography sylva_prior Amazon_employee_access JapaneseVowels rl \
thyroid_sick oil"

echo "===== FASE B: Optuna vs grid — $(date) ====="
# repartir el subconjunto en W grupos y lanzar en paralelo, seeds 0,1
i=0; : > /tmp/cmds_B.sh
declare -a GB
for d in $SUBSET; do GB[$((i%W))]="${GB[$((i%W))]:-} $d"; i=$((i+1)); done
for w in $(seq 0 $((W-1))); do
  [ -z "${GB[$w]:-}" ] && continue
  echo "$PY optuna_study.py ${GB[$w]} --seeds 0,1 --tag optuna --trials 40 > logs/optuna_w$w.log 2>&1" >> /tmp/cmds_B.sh
done
run_parallel /tmp/cmds_B.sh
echo "===== FASE B COMPLETA — $(date) ====="

echo "===== FASE C: familias+ensembles — $(date) ====="
i=0; : > /tmp/cmds_C.sh
declare -a GC
for d in $SUBSET; do GC[$((i%W))]="${GC[$((i%W))]:-} $d"; i=$((i+1)); done
for w in $(seq 0 $((W-1))); do
  [ -z "${GC[$w]:-}" ] && continue
  echo "$PY families_study.py ${GC[$w]} --seeds 0,1 --sizes 1200,5000,20000 --tag families > logs/families_w$w.log 2>&1" >> /tmp/cmds_C.sh
done
run_parallel /tmp/cmds_C.sh
echo "===== FASE C COMPLETA — TODO HECHO $(date) ====="
touch /home/agomez/proyectos/precision-at-k-study/valsplit_study/ALL_DONE
