"""Pong window: you (right, blue) against the CPU (left, red).

    python -m games.pong.play                    # play
    python -m games.pong.play --agente seguidor  # watch the rule-based agent play for you
    python -m games.pong.play --pontos 3         # shorter match
    python -m games.pong.play --nivel dificil    # facil | normal | dificil
    SDL_VIDEODRIVER=dummy python -m games.pong.play --frames 120   # headless smoke test

Keys: Up/W, Down/S move · Space start · P pause · R restart · Esc quit.
All game rules live in env.py; this file only reads the keyboard and draws.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pygame as pg

from games.pong.env import PongEnv, follow_ball_policy

ASSETS = Path(__file__).resolve().parent / "assets"
FPS = 60
AGENTS = {"seguidor": follow_ball_policy}
LEVELS = {"facil": 5.0, "normal": 6.0, "dificil": 7.0}   # CPU paddle speed

BLUE = (40, 110, 230)
RED = (220, 50, 50)
WHITE = (245, 245, 245)
SHADOW = (0, 0, 0, 140)


def keyboard_action():
    keys = pg.key.get_pressed()
    up = keys[pg.K_UP] or keys[pg.K_w]
    down = keys[pg.K_DOWN] or keys[pg.K_s]
    if up and not down:
        return PongEnv.UP
    if down and not up:
        return PongEnv.DOWN
    return PongEnv.STAY


class PongWindow:
    def __init__(self, env: PongEnv, agent=None, agent_name=None):
        self.env = env
        self.agent = agent
        self.agent_name = agent_name or (getattr(agent, "__name__", "agente") if agent else None)
        c = env.cfg
        pg.display.init()
        pg.font.init()
        self.screen = pg.display.set_mode((c.width, c.height))
        pg.display.set_caption("Pong")
        icon = ASSETS / "logo_mini.jpg"
        if icon.exists():
            pg.display.set_icon(pg.image.load(str(icon)))
        bg = ASSETS / "fundo.jpg"
        self.background = None
        if bg.exists():
            self.background = pg.transform.smoothscale(pg.image.load(str(bg)).convert(), (c.width, c.height))
        self.font_score = pg.font.Font(None, 96)
        self.font_big = pg.font.Font(None, 72)
        self.font_small = pg.font.Font(None, 30)
        self.clock = pg.time.Clock()
        self.state = "start"          # start -> playing <-> paused -> over

    # ------------------------------------------------------------- loop
    def run(self, max_frames: int | None = None):
        frames = 0
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and self.handle_key(event.key) == "quit":
                    return
            if self.state == "playing":
                obs = self.env.observation()
                action = self.agent(obs) if self.agent else keyboard_action()
                _, _, done, _ = self.env.step(action)
                if done:
                    self.state = "over"
            self.draw()
            pg.display.flip()
            self.clock.tick(FPS)
            frames += 1
            if max_frames is not None:
                if frames == 1 and self.state == "start":
                    self.state = "playing"     # smoke test: skip the start screen
                if frames >= max_frames:
                    return

    def handle_key(self, key):
        if key == pg.K_ESCAPE:
            return "quit"
        if key == pg.K_r:
            self.env.reset()
            self.state = "playing"
        elif key == pg.K_SPACE and self.state in ("start", "over"):
            if self.state == "over":
                self.env.reset()
            self.state = "playing"
        elif key == pg.K_p and self.state in ("playing", "paused"):
            self.state = "paused" if self.state == "playing" else "playing"
        return None

    # ------------------------------------------------------------- drawing
    def draw(self):
        env, c, scr = self.env, self.env.cfg, self.screen
        if self.background is not None:
            scr.blit(self.background, (0, 0))
        else:
            scr.fill((34, 140, 70))
            pg.draw.rect(scr, WHITE, (c.width // 2 - 1, 0, 2, c.height))
        pg.draw.rect(scr, WHITE, (0, 0, c.width, c.wall))
        pg.draw.rect(scr, WHITE, (0, c.height - c.wall, c.width, c.wall))

        pg.draw.rect(scr, RED, (env.cpu_x, env.cpu_y, c.paddle_w, c.paddle_h), border_radius=3)
        pg.draw.rect(scr, BLUE, (env.player_x, env.player_y, c.paddle_w, c.paddle_h), border_radius=3)
        pg.draw.ellipse(scr, WHITE, (env.ball_x, env.ball_y, c.ball_size, c.ball_size))

        self.text(str(env.score_cpu), self.font_score, (c.width // 2 - 80, 60), RED)
        self.text(str(env.score_player), self.font_score, (c.width // 2 + 80, 60), BLUE)
        who = f"agente {self.agent_name}" if self.agent else "você"
        self.text(f"CPU  x  {who}   ·   primeiro a {c.points_to_win} pontos", self.font_small,
                  (c.width // 2, c.height - 25), WHITE)

        if self.state == "start":
            self.overlay("PONG", "Espaço para começar",
                         "Setas ou W/S movem a raquete azul · P pausa · R reinicia · Esc sai")
        elif self.state == "paused":
            self.overlay("Pausado", "P para continuar")
        elif self.state == "over":
            title = "Você venceu!" if env.winner == "player" else (
                "A CPU venceu!" if env.winner == "cpu" else "Fim de jogo")
            self.overlay(title, f"{env.score_cpu} x {env.score_player}",
                         "Espaço ou R para jogar de novo · Esc sai")

    def overlay(self, title, line1, line2=None):
        c = self.env.cfg
        shade = pg.Surface((c.width, c.height), pg.SRCALPHA)
        shade.fill(SHADOW)
        self.screen.blit(shade, (0, 0))
        panel = pg.Rect(0, 0, 760, 200 if line2 else 150)
        panel.center = (c.width // 2, c.height // 2)
        pg.draw.rect(self.screen, (15, 40, 25), panel, border_radius=16)
        pg.draw.rect(self.screen, WHITE, panel, width=2, border_radius=16)
        self.text(title, self.font_big, (c.width // 2, c.height // 2 - 50), WHITE)
        self.text(line1, self.font_small, (c.width // 2, c.height // 2 + 10), WHITE)
        if line2:
            self.text(line2, self.font_small, (c.width // 2, c.height // 2 + 50), WHITE)

    def text(self, s, font, center, color):
        surf = font.render(s, True, color)
        self.screen.blit(surf, surf.get_rect(center=center))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pong: você contra a CPU.")
    ap.add_argument("--pontos", type=int, default=7, help="pontos para vencer (padrão 7)")
    ap.add_argument("--nivel", choices=list(LEVELS), default="normal", help="dificuldade da CPU")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agente", choices=sorted(AGENTS), help="deixa um agente jogar no seu lugar")
    ap.add_argument("--frames", type=int, default=None, help="sai após N quadros (teste sem tela)")
    args = ap.parse_args(argv)

    env = PongEnv(seed=args.seed, points_to_win=args.pontos, cpu_speed=LEVELS[args.nivel])
    PongWindow(env, agent=AGENTS.get(args.agente), agent_name=args.agente).run(max_frames=args.frames)
    pg.quit()


if __name__ == "__main__":
    sys.exit(main())
