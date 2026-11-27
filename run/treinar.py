"""Train agents with DQN and save the learning curves in data/experimentos/<game>/<agent>/.

    python run/treinar.py --jogo pong --agente vqc --sementes 0 1 2 3 4
    python run/treinar.py --jogo snake --agente mlp cv --sementes 0 --passos 5000   # quick test

Settings come from src/experimentos.py (same for every agent: fair comparison).
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
import dataclasses
import time

from experimentos import AGENTS, GAMES, make_agent
from paths import DATA
from rl.dqn import train


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jogo", choices=list(GAMES), required=True)
    ap.add_argument("--agente", choices=AGENTS, nargs="+", required=True)
    ap.add_argument("--sementes", type=int, nargs="+", default=[0])
    ap.add_argument("--passos", type=int, default=None, help="override total_steps (quick tests)")
    ap.add_argument("--saida", default=None, help="output folder (default data/experimentos)")
    args = ap.parse_args()

    g = GAMES[args.jogo]
    cfg = g["dqn"]
    if args.passos:
        cfg = dataclasses.replace(cfg, total_steps=args.passos,
                                  eps_decay_steps=min(cfg.eps_decay_steps, args.passos // 2),
                                  warmup=min(cfg.warmup, args.passos // 5))
    out = _Path(args.saida) if args.saida else DATA / "experimentos"
    for agent in args.agente:
        for seed in args.sementes:
            t = time.perf_counter()
            path = out / args.jogo / agent / f"seed_{seed}.csv"
            _, hist = train(g["env"], make_agent(agent, args.jogo), cfg, seed=seed, log_path=path)
            last = hist[-10:]
            mean = sum(h["return"] for h in last) / max(1, len(last))
            print(f"{args.jogo:5s} {agent:9s} seed {seed}: {len(hist)} episodes, "
                  f"mean return (last 10) {mean:.2f}, {time.perf_counter() - t:.0f}s -> {path}")


if __name__ == "__main__":
    main()
