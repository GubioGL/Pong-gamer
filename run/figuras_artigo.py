"""Figures of the paper draft.

    python run/figuras_artigo.py

REAL (from the code / measured):
    fig_jogos.pdf            the two environments, as rendered by games/*/play.py
    fig_circuito_qubits.pdf  one layer of the qubit VQC (PennyLane draw_mpl)
    fig_circuito_cv.pdf      one layer of the Gaussian CV QNN (PennyLane draw_mpl)
    fig_custo.pdf            parameters and time per DQN gradient step (data/custo/, run/medir_custo.py)
ILLUSTRATIVE (synthetic curves, NOT results; they show the layout and the comparison we want):
    fig_curvas_ilustrativa.pdf      learning curves, mean +- std over seeds
    fig_parametros_ilustrativa.pdf  final return versus number of parameters
Replace the illustrative ones with run/figuras_resultados.py once data/experimentos/ is filled.
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

_os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
_os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import DATA, save_figure

# Fixed categorical order (validated: CVD-safe; contrast WARN -> every series also has
# a direct label, its own line style and marker).
AGENTS = ["mlp", "mlp_igual", "vqc", "cv"]
LABEL = {"mlp": "MLP 2x64", "mlp_igual": "MLP (same #params as VQC)",
         "vqc": "Qubit VQC", "cv": "CV QNN (Gaussian)"}
SHORT = {"mlp": "MLP 2x64", "mlp_igual": "MLP small", "vqc": "Qubit VQC", "cv": "CV QNN"}
COLOR = {"mlp": "#257de0", "mlp_igual": "#f47230", "vqc": "#00b894", "cv": "#f2a600"}
STYLE = {"mlp": "-", "mlp_igual": "--", "vqc": "-.", "cv": ":"}
MARKER = {"mlp": "o", "mlp_igual": "s", "vqc": "^", "cv": "D"}
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
})


def spread_labels(ys, min_gap):
    """Push apart label y-positions that are closer than min_gap (keeps their order)."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < min_gap:
            out[j] = out[i] + min_gap
    return out


def bottom_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))


def watermark(fig):
    fig.text(0.5, 0.5, "ILLUSTRATIVE - synthetic, not results", ha="center", va="center",
             rotation=18, fontsize=17, color="#c8c8c8", alpha=0.55, zorder=0, weight="bold")


# ------------------------------------------------------------------ real figures
def fig_games():
    import pygame as pg
    from games.pong.env import PongEnv, follow_ball_policy
    from games.pong.play import PongWindow
    from games.snake.env import SnakeEnv, greedy_policy
    from games.snake.play import SnakeWindow

    pong = PongEnv(seed=3)
    pw = PongWindow(pong, sound=False)
    for _ in range(260):
        pong.step(follow_ball_policy(pong.observation()))
    pong.score_cpu, pong.score_player = 1, 2
    pw.state = "playing"
    pw.draw()
    img_pong = pg.surfarray.array3d(pw.screen).swapaxes(0, 1)

    snake = SnakeEnv(seed=5)
    sw = SnakeWindow(snake, sound=False)
    obs = snake.reset()
    for _ in range(90):
        obs, _, done, _ = snake.step(greedy_policy(obs))
        if done:
            break
    sw.best, sw.state = snake.score, "playing"
    sw.draw()
    img_snake = pg.surfarray.array3d(sw.screen).swapaxes(0, 1)
    pg.quit()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, img, title, txt in [
        (axes[0], img_pong, "(a) Pong", "obs: ball x, y, vx, vy; paddles y (6)\nactions: stay, up, down"),
        (axes[1], img_snake, "(b) Snake, 12x12", "obs: food dx, dy; danger x4; direction (8)\nactions: up, down, left, right"),
    ]:
        ax.imshow(img)
        ax.set_title(title, loc="left")
        ax.set_xlabel(txt, color=INK2)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    return save_figure(fig, "fig_jogos")


def fig_circuits():
    import jax
    from agentes.cvqnn import CVQNet
    from agentes.vqc import QubitQNet

    q = QubitQNet(6, 3, layers=1)
    fig, _ = q.draw(q.init(jax.random.PRNGKey(0)), decimals=None, style="black_white", fontsize=13)
    p1 = save_figure(fig, "fig_circuito_qubits")
    plt.close(fig)

    c = CVQNet(6, 3, layers=1)
    fig, _ = c.draw(c.init(jax.random.PRNGKey(0)), decimals=None, style="black_white", fontsize=13)
    p2 = save_figure(fig, "fig_circuito_cv")
    plt.close(fig)
    return p1, p2


def fig_cost():
    path = DATA / "custo" / "custo_por_agente.csv"
    rows = list(csv.DictReader(path.open()))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    width = 0.2
    games = ["pong", "snake"]
    for ax, key, ylabel, log in [(axes[0], "parameters", "trainable parameters", True),
                                 (axes[1], "step_ms", "ms per gradient step (batch 32, CPU)", True)]:
        for k, agent in enumerate(AGENTS):
            vals = [float(next(r[key] for r in rows if r["game"] == g and r["agent"] == agent)) for g in games]
            xs = np.arange(len(games)) + (k - 1.5) * width
            bars = ax.bar(xs, vals, width * 0.9, color=COLOR[agent], label=LABEL[agent],
                          hatch=["", "///", "...", "xx"][k], edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.12, (f"{v:.0f}" if v >= 10 else f"{v:.2g}"), ha="center", fontsize=6, color=INK)
        ax.set_xticks(range(len(games)), ["Pong", "Snake"])
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(top=2e4)
    axes[1].set_ylim(top=100)
    bottom_legend(fig, axes[0])
    return save_figure(fig, "fig_custo")


# ------------------------------------------------------------ illustrative figures
def _synthetic(final, rise, spread, start, x, rng, seeds=10):
    """Logistic learning curve + seed-to-seed spread. Purely illustrative."""
    curves = []
    for _ in range(seeds):
        f = final + rng.normal(0, spread)
        r = rise * rng.uniform(0.8, 1.25)
        y = start + (f - start) / (1 + np.exp(-(x - r) / (0.18 * r)))
        y = y + rng.normal(0, spread * 0.25, x.size).cumsum() * 0.05
        curves.append(y)
    c = np.array(curves)
    return c.mean(0), c.std(0)


# Shapes inspired by the literature (skolik2022gym, franz2022instab, nagy2021photonic):
# small quantum models comparable to a classical model of the same size, larger spread.
HYP = {
    "pong": {"mlp": (2.6, 30, 0.2), "mlp_igual": (1.7, 45, 0.35), "vqc": (1.9, 50, 0.6), "cv": (1.4, 55, 0.7)},
    "snake": {"mlp": (18, 35, 1.5), "mlp_igual": (11, 45, 2.0), "vqc": (12, 50, 3.5), "cv": (9, 55, 3.5)},
}
START = {"pong": -2.8, "snake": -0.8}


def fig_curves_illustrative():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 200)          # thousands of environment steps
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, game, ylabel in [(axes[0], "pong", "return per match (3 points)"),
                             (axes[1], "snake", "return per episode (food eaten)")]:
        ends = {}
        for agent in AGENTS:
            final, rise, spread = HYP[game][agent]
            m, s = _synthetic(final, rise, spread, START[game], x, rng)
            ax.fill_between(x, m - s, m + s, color=COLOR[agent], alpha=0.15, linewidth=0)
            ax.plot(x, m, STYLE[agent], color=COLOR[agent], label=LABEL[agent])
            ax.plot(x[-1], m[-1], MARKER[agent], color=COLOR[agent], ms=4)
            ends[agent] = m[-1]
        span = np.ptp(ax.get_ylim())
        ys = spread_labels([ends[a] for a in AGENTS], 0.07 * span)
        for agent, y in zip(AGENTS, ys):
            ax.annotate(SHORT[agent], (x[-1], ends[agent]), xytext=(x[-1] + 3, y), textcoords="data",
                        va="center", fontsize=6.5, color=INK)
        ax.set_title(f"({'a' if game == 'pong' else 'b'}) {game.capitalize()}", loc="left")
        ax.set_xlabel("environment steps (x1000)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 125)
    bottom_legend(fig, axes[0])
    watermark(fig)
    return save_figure(fig, "fig_curvas_ilustrativa")


def fig_params_illustrative():
    rng = np.random.default_rng(1)
    params = {"pong": {"mlp": 4803, "mlp_igual": 93, "vqc": 90, "cv": 166},
              "snake": {"mlp": 4996, "mlp_igual": 121, "vqc": 120, "cv": 224}}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, game in zip(axes, ["pong", "snake"]):
        for agent in AGENTS:
            final, _, spread = HYP[game][agent]
            vals = final + rng.normal(0, spread, 10)
            ax.errorbar(params[game][agent], vals.mean(), yerr=vals.std(), fmt=MARKER[agent],
                        color=COLOR[agent], ms=6, capsize=3, label=LABEL[agent])
            dx, dy = {"mlp": (8, 4), "mlp_igual": (-8, -12), "vqc": (-8, 8), "cv": (8, -2)}[agent]
            ax.annotate(SHORT[agent], (params[game][agent], vals.mean()), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.5, color=INK,
                        ha="right" if dx < 0 else "left")
        ax.set_xscale("log")
        ax.set_xlim(25, 2e4)
        ax.set_xlabel("trainable parameters (log)")
        ax.set_ylabel("final greedy return")
        ax.set_title(f"({'a' if game == 'pong' else 'b'}) {game.capitalize()}", loc="left")
    bottom_legend(fig, axes[0])
    watermark(fig)
    return save_figure(fig, "fig_parametros_ilustrativa")


if __name__ == "__main__":
    for f in (fig_games, fig_circuits, fig_cost, fig_curves_illustrative, fig_params_illustrative):
        out = f()
        plt.close("all")
        print(f.__name__, "->", out)
