#!/usr/bin/env bash
# Watchdog: relanza el experimento HPO extra (hyperband/bohb/gp) si no está corriendo
# y aún no ha terminado. Idempotente: seguro llamarlo cada pocos minutos.
STUDY=/home/agomez/proyectos/precision-at-k-study/valsplit_study
PY=/home/agomez/proyectos/precision-at-k-study/benchmark/env/bin/python
LOG=$STUDY/logs/hpo_watchdog.log
DONE=$STUDY/logs/hpo_extra.done
cd "$STUDY" || exit 0
mkdir -p logs

ts() { date +'%Y-%m-%d %H:%M:%S'; }

# ¿ya terminado? (32 celdas para cada uno de hyperband/bohb/gp)
complete=$("$PY" - <<'PY'
import glob, json
from collections import Counter
c=Counter()
for f in glob.glob("results/hpo__*.json"):
    for r in json.load(open(f)):
        if r.get("budget")==160 and r.get("method") in ("hyperband","bohb","gp"): c[r["method"]]+=1
print(1 if all(c[m]>=32 for m in ("hyperband","bohb","gp")) else 0, c["hyperband"], c["bohb"], c["gp"])
PY
)
set -- $complete
if [ "$1" = "1" ]; then
    echo "$(ts) COMPLETO (hb=$2 bohb=$3 gp=$4). Nada que hacer." >> "$LOG"
    touch "$DONE"
    # auto-limpieza del cron
    (crontab -l 2>/dev/null | grep -v 'hpo_watchdog.sh') | crontab - 2>/dev/null
    exit 0
fi

# ¿ya hay un proceso corriendo?
if pgrep -f "hpo_mf.py|hpo_study.py .*--methods gp" >/dev/null 2>&1; then
    echo "$(ts) ya corriendo (hb=$2 bohb=$3 gp=$4). OK." >> "$LOG"
    exit 0
fi

# no corre y no está completo -> relanzar (detached, autocontenido)
echo "$(ts) CAÍDO (hb=$2 bohb=$3 gp=$4) -> relanzo experimento" >> "$LOG"
DS="jm1 online_shoppers_intention Bank_marketing_data_set_UCI Pulsar-Dataset-HTRU2 letter satimage mammography rl"
setsid bash -c "cd $STUDY && \
  $PY hpo_mf.py $DS --seeds 0,1 --fracs 0.3,1.0 >> logs/hpo_mf.log 2>&1 && \
  $PY hpo_study.py $DS --seeds 0,1 --fracs 0.3,1.0 --methods gp >> logs/hpo_gp.log 2>&1" >> "$LOG" 2>&1 &
exit 0
