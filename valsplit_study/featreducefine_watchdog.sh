#!/usr/bin/env bash
STUDY=/home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=/home/agomez/proyectos/precision-at-k-study/benchmark/env/bin/python
LOG=$STUDY/logs/featreducefine_watchdog.log; cd "$STUDY" || exit 0; mkdir -p logs
ts(){ date +'%Y-%m-%d %H:%M:%S'; }
n=$("$PY" -c "import glob,json;print(sum(1 for f in glob.glob('results/featreducefine__*.json') for r in json.load(open(f))))" 2>/dev/null)
if [ "${n:-0}" -ge 352 ]; then
  echo "$(ts) COMPLETO ($n/352). Limpio cron." >> "$LOG"
  (crontab -l 2>/dev/null | grep -v 'featreducefine_watchdog.sh') | crontab - 2>/dev/null; exit 0
fi
if pgrep -f "feat_reduce.py .*featreducefine" >/dev/null 2>&1; then echo "$(ts) ya corriendo ($n/352)" >> "$LOG"; exit 0; fi
echo "$(ts) CAIDO ($n/352) -> relanzo" >> "$LOG"
setsid bash -c "cd $STUDY && $PY feat_reduce.py jm1 online_shoppers_intention Pulsar-Dataset-HTRU2 letter satimage mammography coil_2000 pendigits --seeds 0,1 --tag featreducefine --thrs 0.900,0.905,0.910,0.915,0.920,0.925,0.930,0.935,0.940,0.945,0.950,0.955,0.960,0.965,0.970,0.975,0.980,0.985,0.990,0.995,1.000 >> logs/featreducefine.log 2>&1" >> "$LOG" 2>&1 &
