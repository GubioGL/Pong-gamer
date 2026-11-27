"""Snake window, same retro arcade look as Pong.

    python -m games.snake.play                     # play (20x15 grid)
    python -m games.snake.play --grade 12x12       # the grid used in the experiments
    python -m games.snake.play --nivel dificil     # facil | normal | dificil (speed)
    python -m games.snake.play --agente guloso     # watch the rule-based agent
    SDL_VIDEODRIVER=dummy python -m games.snake.play --frames 120   # headless smoke test

Keys: arrows or WASD · Space start · P pause · R restart · M sound · Esc quit.
All game rules live in env.py; this file only reads the keyboard, draws and beeps.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque

import pygame as pg

from games import retro
from games.snake.env import SnakeEnv, greedy_policy

FPS = 60
CELL = 36
HEADER = 110
MARGIN = 20
AGENTS = {"guloso": greedy_policy}
LEVELS = {"facil": 7, "normal": 10, "dificil": 14}      # snake moves per second
KEYMAP = {pg.K_UP: SnakeEnv.UP, pg.K_w: SnakeEnv.UP, pg.K_DOWN: SnakeEnv.DOWN, pg.K_s: SnakeEnv.DOWN,
          pg.K_LEFT: SnakeEnv.LEFT, pg.K_a: SnakeEnv.LEFT, pg.K_RIGHT: SnakeEnv.RIGHT, pg.K_d: SnakeEnv.RIGHT}


class SnakeWindow:
    def __init__(self, env: SnakeEnv, agent=None, agent_name=None, moves_per_second=10, sound=True):
        self.env = env
        self.agent = agent
        self.agent_name = agent_name
        c = env.cfg
        self.board = pg.Rect(MARGIN, HEADER, c.grid_w * CELL, c.grid_h * CELL)
        pg.display.init()
        self.screen = pg.display.set_mode((self.board.width + 2 * MARGIN, self.board.bottom + MARGIN + 30))
        pg.display.set_caption("Cobrinha")
        self.fonts = retro.Fonts()
        self.beeper = retro.Beeper(enabled=sound)
        self.sound_on = sound
        self.clock = pg.time.Clock()
        self.frames_per_move = max(1, round(FPS / moves_per_second))
        self.pending = deque(maxlen=3)          # buffered key presses: quick turns are not lost
        self.best = 0
        self.state = "start"

    def run(self, max_frames: int | None = None):
        frames = 0
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and self.handle_key(event.key) == "quit":
                    return
            if self.state == "playing" and frames % self.frames_per_move == 0:
                self.advance()
            self.draw()
            pg.display.flip()
            self.clock.tick(FPS)
            frames += 1
            if max_frames is not None:
                if frames == 1 and self.state == "start":
                    self.state = "playing"
                if frames >= max_frames:
                    return

    def advance(self):
        env = self.env
        if self.agent:
            action = self.agent(env.observation())
        elif self.pending:
            action = self.pending.popleft()
        else:
            action = env.direction
        _, _, done, info = env.step(action)
        self.best = max(self.best, env.score)
        if self.sound_on and info["event"] == "eat":
            self.beeper.play("eat")
        if done:
            if self.sound_on and info["event"] in ("wall", "self"):
                self.beeper.play("dead")
            self.state = "over"

    def handle_key(self, key):
        if key == pg.K_ESCAPE:
            return "quit"
        if key in KEYMAP and self.state == "playing":
            self.pending.append(KEYMAP[key])
        elif key == pg.K_r:
            self.restart()
        elif key == pg.K_m:
            self.sound_on = not self.sound_on
        elif key == pg.K_SPACE and self.state in ("start", "over"):
            if self.state == "over":
                self.restart()
            self.state = "playing"
        elif key == pg.K_p and self.state in ("playing", "paused"):
            self.state = "paused" if self.state == "playing" else "playing"
        return None

    def restart(self):
        self.env.reset()
        self.pending.clear()
        self.state = "playing"

    # ------------------------------------------------------------- drawing
    def cell_rect(self, x, y, inset=2):
        return pg.Rect(self.board.x + x * CELL + inset, self.board.y + y * CELL + inset,
                       CELL - 2 * inset, CELL - 2 * inset)

    def draw(self):
        env, scr = self.env, self.screen
        w, _ = scr.get_size()
        scr.fill(retro.BLACK)
        retro.draw_number(scr, env.score, w // 2 - 120, 22, block=12)
        retro.draw_number(scr, self.best, w // 2 + 120, 22, block=12, color=retro.GREY)
        self.fonts.text(scr, "PONTOS", self.fonts.small, (w // 2 - 120, 98), retro.GREY)
        self.fonts.text(scr, "RECORDE", self.fonts.small, (w // 2 + 120, 98), retro.GREY)
        pg.draw.rect(scr, retro.WHITE, self.board.inflate(8, 8), width=4)

        fx, fy = env.food
        pg.draw.rect(scr, retro.WHITE, self.cell_rect(fx, fy, inset=9))
        for i, (x, y) in enumerate(env.body):
            r = self.cell_rect(x, y)
            pg.draw.rect(scr, retro.WHITE, r)
            if i == 0:                                   # head: dark "eye" in the moving direction
                dx, dy = SnakeEnv.DIRS[env.direction]
                eye = pg.Rect(0, 0, 8, 8)
                eye.center = (r.centerx + dx * 8, r.centery + dy * 8)
                pg.draw.rect(scr, retro.BLACK, eye)

        who = f"AGENTE {self.agent_name.upper()}" if self.agent else "VOCÊ"
        c = env.cfg
        self.fonts.text(scr, f"{who}   -   GRADE {c.grid_w}x{c.grid_h}", self.fonts.small,
                        (w // 2, self.board.bottom + MARGIN + 8), retro.GREY)

        if self.state == "start":
            retro.overlay(scr, self.fonts, "Cobrinha", "ESPAÇO para começar",
                          "Setas/WASD - P pausa - R reinicia - Esc sai")
        elif self.state == "paused":
            retro.overlay(scr, self.fonts, "Pausado", "P para continuar")
        elif self.state == "over":
            why = {"wall": "Bateu na parede", "self": "Mordeu o próprio rabo",
                   "idle": "Tempo esgotado"}.get(env.last_event, "Fim de jogo")
            retro.overlay(scr, self.fonts, "Fim de jogo", f"{why} - {env.score} pontos",
                          "ESPAÇO ou R para jogar de novo - Esc sai")


def parse_grid(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("use LxA, por exemplo 20x15")
    if w < 5 or h < 5:
        raise argparse.ArgumentTypeError("mínimo 5x5")
    return w, h


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cobrinha (Snake).")
    ap.add_argument("--grade", type=parse_grid, default=(20, 15), help="tamanho LxA (padrão 20x15)")
    ap.add_argument("--nivel", choices=list(LEVELS), default="normal", help="velocidade")
    ap.add_argument("--sem-som", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agente", choices=sorted(AGENTS), help="deixa um agente jogar")
    ap.add_argument("--frames", type=int, default=None, help="sai após N quadros (teste sem tela)")
    args = ap.parse_args(argv)

    gw, gh = args.grade
    # humans: no idle limit in practice; agents keep the experiment default
    env = SnakeEnv(seed=args.seed, grid_w=gw, grid_h=gh, max_idle=10 ** 9 if not args.agente else 200,
                   max_steps=10 ** 9)
    window = SnakeWindow(env, agent=AGENTS.get(args.agente), agent_name=args.agente,
                         moves_per_second=LEVELS[args.nivel],
                         sound=not args.sem_som and args.frames is None)
    window.run(max_frames=args.frames)
    pg.quit()


if __name__ == "__main__":
    sys.exit(main())
