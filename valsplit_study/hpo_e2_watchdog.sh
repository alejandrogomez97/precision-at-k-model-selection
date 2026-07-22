#!/usr/bin/env bash
# Watchdog del run HPO en marco E2 (tag hpoe2). Relanza si se cae; se autolimpia al terminar.
STUDY=/home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=/home/agomez/proyectos/precision-at-k-study/benchmark/env/bin/python
LOG=$STUDY/logs/hpo_e2_watchdog.log
cd "$STUDY" || exit 0; mkdir -p logs
ts() { date +'%Y-%m-%d %H:%M:%S'; }
DS="jm1 online_shoppers_intention Bank_marketing_data_set_UCI Pulsar-Dataset-HTRU2 letter satimage mammography rl"

st=$("$PY" - <<'PY'
import glob, json
from collections import Counter
c=Counter()
for f in glob.glob("results/hpoe2__*.json"):
    for r in json.load(open(f)):
        if r.get("budget")==160: c[r["method"]]+=1
M=["grid","random","tpe","cmaes","gp","hyperband","bohb"]
print(1 if all(c[m]>=32 for m in M) else 0, min(c[m] for m in M) if c else 0)
PY
)
set -- $st
if [ "$1" = "1" ]; then
    echo "$(ts) COMPLETO (min=$2/32). Limpio cron." >> "$LOG"
    (crontab -l 2>/dev/null | grep -v 'hpo_e2_watchdog.sh') | crontab - 2>/dev/null
    exit 0
fi
if pgrep -f "hpo_e2.py" >/dev/null 2>&1; then
    echo "$(ts) ya corriendo (min=$2/32). OK." >> "$LOG"; exit 0
fi
echo "$(ts) CAÍDO (min=$2/32) -> relanzo" >> "$LOG"
setsid bash -c "cd $STUDY && $PY hpo_e2.py $DS --seeds 0,1 --fracs 0.3,1.0 >> logs/hpo_e2.log 2>&1" >> "$LOG" 2>&1 &
exit 0
