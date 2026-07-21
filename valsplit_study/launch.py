"""Reparte los datasets en W grupos round-robin (ordenados por tamaño para
balancear coste) y lanza W procesos en paralelo. Uso:
    python launch.py <tag> <n_workers> [--seeds a,b,c] [--sizes ...]
Imprime los comandos; el .sh de arranque los ejecuta en background.
"""
import sys, json, subprocess, os
sys.path.insert(0, "/home/agomez/proyectos/precision-at-k-study/benchmark")
import common as C
import pandas as pd

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "grid"
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    extra = sys.argv[3:]
    names = C.list_materialized()
    # ordenar por nº de filas (coste) para repartir grandes entre workers
    eda = pd.read_csv(f"{C.BDIR}/eda_summary.csv").set_index("name")
    def nrows(nm):
        try: return int(eda.loc[nm, "n"])
        except Exception: return 0
    names = sorted(names, key=nrows, reverse=True)
    groups = [[] for _ in range(W)]
    for i, nm in enumerate(names):
        groups[i % W].append(nm)
    # cada worker procesa PRIMERO los pequeños (resultados rápidos, breadth-first);
    # los grandes al final
    for g in groups:
        g.sort(key=nrows)
    os.makedirs(f"{STUDY}/logs", exist_ok=True)
    cmds = []
    for w, g in enumerate(groups):
        log = f"{STUDY}/logs/{tag}_w{w}.log"
        cmd = (f"{C.BDIR}/env/bin/python {STUDY}/core.py " + " ".join(g) +
               f" --tag {tag} " + " ".join(extra) + f" > {log} 2>&1")
        cmds.append(cmd)
    print("\n".join(cmds))

if __name__ == "__main__":
    main()
