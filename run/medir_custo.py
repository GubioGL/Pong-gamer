"""Measure the cost of each agent: trainable parameters and time per DQN gradient step (CPU).

    python run/medir_custo.py          # -> data/custo/custo_por_agente.csv

Real measurement (not a learning result): used by the cost figure of the paper.
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

import csv
import platform
import time

import jax
import numpy as np

from experimentos import AGENTS, GAMES, make_agent
from paths import DATA
from rl.dqn import count_parameters, make_functions, make_optimizer

REPEATS = 30


def main():
    rows = []
    for game, g in GAMES.items():
        cfg = g["dqn"]
        n, b = g["n_obs"], cfg.batch_size
        for name in AGENTS:
            model = make_agent(name, game)
            params = model.init(jax.random.PRNGKey(0))
            opt = make_optimizer(model, params, cfg)
            state = opt.init(params)
            act, update = make_functions(model, opt, cfg)
            o = np.random.default_rng(0).uniform(-1, 1, (b, n)).astype(np.float32)
            a = np.zeros(b, np.int32)
            r = np.zeros(b, np.float32)
            t = time.perf_counter()
            jax.block_until_ready(update(params, params, state, o, a, r, o, r))
            compile_s = time.perf_counter() - t
            t = time.perf_counter()
            for _ in range(REPEATS):
                jax.block_until_ready(update(params, params, state, o, a, r, o, r))
            step_ms = (time.perf_counter() - t) / REPEATS * 1000
            rows.append({"game": game, "agent": name, "parameters": count_parameters(params),
                         "step_ms": round(step_ms, 3), "compile_s": round(compile_s, 1),
                         "batch": b, "cpu": platform.processor() or platform.machine()})
            print(rows[-1])
    out = DATA / "custo" / "custo_por_agente.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", out.relative_to(_ROOT))


if __name__ == "__main__":
    main()
