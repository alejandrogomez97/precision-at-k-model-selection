#!/usr/bin/env bash
# Watchdog del experimento de reducción por correlación (Parte 4). Relanza si se cae;
# se autolimpia al completar (8 datasets × 2 seeds × 7 niveles = 112 registros).
STUDY=/home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=/home/agomez/proyectos/precision-at-k-study/benchmark/env/bin/python
LOG=$STUDY/logs/feat_reduce_watchdog.log
cd "$STUDY" || exit 0; mkdir -p logs
ts() { date +'%Y-%m-%d %H:%M:%S'; }
DS="jm1 online_shoppers_intention Pulsar-Dataset-HTRU2 letter satimage mammography coil_2000 pendigits"

n=$("$PY" - <<'PY'
import glob, json
print(sum(1 for f in glob.glob("results/featreduce__*.json") for r in json.load(open(f))))
PY
)
if [ "${n:-0}" -ge 112 ]; then
    echo "$(ts) COMPLETO ($n/112). Limpio cron." >> "$LOG"
    (crontab -l 2>/dev/null | grep -v 'feat_reduce_watchdog.sh') | crontab - 2>/dev/null
    exit 0
fi
if pgrep -f "feat_reduce.py" >/dev/null 2>&1; then
    echo "$(ts) ya corriendo ($n/112). OK." >> "$LOG"; exit 0
fi
echo "$(ts) CAÍDO ($n/112) -> relanzo" >> "$LOG"
setsid bash -c "cd $STUDY && $PY feat_reduce.py $DS --seeds 0,1 >> logs/feat_reduce.log 2>&1" >> "$LOG" 2>&1 &
exit 0
