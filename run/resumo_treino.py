"""Compact summary of training runs, short enough to paste into a chat.

    python run/resumo_treino.py                       # everything in data/experimentos
    python run/resumo_treino.py teste_treino          # another folder (as used with --saida)
    python run/resumo_treino.py teste_treino --jogo pong --agente mlp

Per game / agent / seed it prints: episodes, mean return of the first and of the last 20% of the
episodes, best episode, final epsilon, loss at the start and at the end, steps per second, plus
the reference scores (random and rule-based player) so a number can be judged.
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root. Keep this block in every script.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _HERE, _ROOT):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import argparse
import csv
import json
import statistics as st

from paths import DATA


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def mean(xs):
    return st.mean(xs) if xs else float("nan")


def reference():
    p = DATA / "referencia" / "baselines.csv"
    if not p.exists():
        return {}
    return {(r["game"], r["policy"]): float(r["mean_return"]) for r in load(p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pasta", nargs="?", default=str(DATA / "experimentos"))
    ap.add_argument("--jogo")
    ap.add_argument("--agente")
    args = ap.parse_args()

    root = _Path(args.pasta)
    ref = reference()
    files = sorted(root.glob("*/*/seed_*.csv"))
    if not files:
        print(f"no seed_*.csv under {root}")
        return
    print(f"folder: {root}")
    last_game = None
    for f in files:
        game, agent = f.parent.parent.name, f.parent.name
        if (args.jogo and game != args.jogo) or (args.agente and agent != args.agente):
            continue
        if game != last_game:
            last_game = game
            print(f"\n== {game}: reference return  random {ref.get((game, 'random'), float('nan')):.2f}"
                  f"  rule-based {ref.get((game, 'rule-based'), float('nan')):.2f}")
            print(f"{'agent':10s} {'seed':>4s} {'eps':>5s} {'first20%':>9s} {'last20%':>9s} {'best':>7s} "
                  f"{'eps_f':>6s} {'loss0':>8s} {'loss1':>8s} {'steps/s':>8s} {'params':>7s}  greedy eval (every 5000 steps)")
        rows = load(f)
        if not rows:
            print(f"{agent:10s} {f.stem[5:]:>4s}  (no finished episode)")
            continue
        ret = [float(r["return"]) for r in rows]
        k = max(1, len(ret) // 5)
        loss = [float(r["loss"]) for r in rows if r["loss"] not in ("", "nan")]
        meta = json.loads(f.with_suffix(".json").read_text()) if f.with_suffix(".json").exists() else {}
        total_steps = int(rows[-1]["env_steps"])
        wall = float(rows[-1]["wall_time"])
        ev = []
        for r in rows:
            if r.get("eval_return") not in (None, "", "None") and (not ev or True):
                v = round(float(r["eval_return"]), 2)
                if not ev or ev[-1][0] != v or ev[-1][1] != r["env_steps"]:
                    ev.append((v, r["env_steps"]))
        seen, evs = set(), []
        for r in rows:                                 # one value per evaluation point
            v = r.get("eval_return")
            if v not in (None, "", "None") and v not in seen:
                seen.add(v)
                evs.append(round(float(v), 1))
        ev_txt = " ".join(f"{v:g}" for v in evs) if evs else "-"
        print(f"{agent:10s} {f.stem[5:]:>4s} {len(ret):5d} {mean(ret[:k]):9.2f} {mean(ret[-k:]):9.2f} {max(ret):7.2f} "
              f"{float(rows[-1]['epsilon']):6.2f} {(loss[0] if loss else float('nan')):8.4f} "
              f"{(loss[-1] if loss else float('nan')):8.4f} {total_steps / max(wall, 1e-9):8.0f} "
              f"{meta.get('parameters', '?'):>7}  {ev_txt}")
    print("\nhow to read: last20% well above first20% (and above 'random') = it is learning; "
          "loss growing a lot = unstable; eps_f > 0.05 means the run ended before epsilon finished decaying.")


if __name__ == "__main__":
    main()
