"""Pong window, retro arcade look: you (right paddle) against the CPU (left paddle).

    python -m games.pong.play                    # play
    python -m games.pong.play --agente seguidor  # watch the rule-based agent play for you
    python -m games.pong.play --pontos 3         # shorter match
    python -m games.pong.play --nivel dificil    # facil | normal | dificil
    python -m games.pong.play --tema campo       # green field instead of the black screen
    SDL_VIDEODRIVER=dummy python -m games.pong.play --frames 120   # headless smoke test

Keys: Up/W, Down/S move · Space start · P pause · R restart · M sound · Esc quit.
All game rules live in env.py; this file only reads the keyboard, draws and beeps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pygame as pg

from games import retro
from games.pong.env import PongEnv, follow_ball_policy

ASSETS = Path(__file__).resolve().parent / "assets"
FPS = 60
AGENTS = {"seguidor": follow_ball_policy}
LEVELS = {"facil": 5.0, "normal": 6.0, "dificil": 7.0}   # CPU paddle speed
SOUND_OF = {"player_hit": "hit", "cpu_hit": "hit", "wall": "wall",
            "point_player": "point", "point_cpu": "point"}


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
    def __init__(self, env: PongEnv, agent=None, agent_name=None, theme="retro", sound=True):
        self.env = env
        self.agent = agent
        self.agent_name = agent_name or (getattr(agent, "__name__", "agente") if agent else None)
        c = env.cfg
        pg.display.init()
        self.screen = pg.display.set_mode((c.width, c.height))
        pg.display.set_caption("Pong")
        icon = ASSETS / "logo_mini.jpg"
        if icon.exists():
            pg.display.set_icon(pg.image.load(str(icon)))
        self.background = None
        if theme == "campo" and (ASSETS / "fundo.jpg").exists():
            img = pg.image.load(str(ASSETS / "fundo.jpg")).convert()
            self.background = pg.transform.smoothscale(img, (c.width, c.height))
        self.fonts = retro.Fonts()
        self.beeper = retro.Beeper(enabled=sound)
        self.sound_on = sound
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
                _, _, done, info = self.env.step(action)
                if self.sound_on and info["event"] in SOUND_OF:
                    self.beeper.play(SOUND_OF[info["event"]])
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
        elif key == pg.K_m:
            self.sound_on = not self.sound_on
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
            scr.fill(retro.BLACK)
            retro.draw_dashed_vline(scr, c.width // 2, c.height)
        pg.draw.rect(scr, retro.WHITE, (0, 0, c.width, c.wall))
        pg.draw.rect(scr, retro.WHITE, (0, c.height - c.wall, c.width, c.wall))

        retro.draw_number(scr, env.score_cpu, c.width // 2 - 160, 40)
        retro.draw_number(scr, env.score_player, c.width // 2 + 160, 40)

        pg.draw.rect(scr, retro.WHITE, (env.cpu_x, env.cpu_y, c.paddle_w, c.paddle_h))
        pg.draw.rect(scr, retro.WHITE, (env.player_x, env.player_y, c.paddle_w, c.paddle_h))
        pg.draw.rect(scr, retro.WHITE, (env.ball_x, env.ball_y, c.ball_size, c.ball_size))   # square ball

        who = f"AGENTE {self.agent_name.upper()}" if self.agent else "VOCÊ"
        self.fonts.text(scr, f"CPU   x   {who}   -   {c.points_to_win} PONTOS", self.fonts.small,
                        (c.width // 2, c.height - 22), retro.GREY)

        if self.state == "start":
            self.overlay("Pong", "ESPAÇO para começar",
                         "Setas ou W/S movem - P pausa - R reinicia - M som - Esc sai")
        elif self.state == "paused":
            self.overlay("Pausado", "P para continuar")
        elif self.state == "over":
            title = "Você venceu!" if env.winner == "player" else (
                "A CPU venceu!" if env.winner == "cpu" else "Fim de jogo")
            self.overlay(title, f"{env.score_cpu}  x  {env.score_player}",
                         "ESPAÇO ou R para jogar de novo - Esc sai")

    def overlay(self, title, line1, line2=None):
        retro.overlay(self.screen, self.fonts, title, line1, line2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pong: você contra a CPU.")
    ap.add_argument("--pontos", type=int, default=7, help="pontos para vencer (padrão 7)")
    ap.add_argument("--nivel", choices=list(LEVELS), default="normal", help="dificuldade da CPU")
    ap.add_argument("--tema", choices=["retro", "campo"], default="retro", help="visual")
    ap.add_argument("--sem-som", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agente", choices=sorted(AGENTS), help="deixa um agente jogar no seu lugar")
    ap.add_argument("--frames", type=int, default=None, help="sai após N quadros (teste sem tela)")
    args = ap.parse_args(argv)

    env = PongEnv(seed=args.seed, points_to_win=args.pontos, cpu_speed=LEVELS[args.nivel])
    window = PongWindow(env, agent=AGENTS.get(args.agente), agent_name=args.agente,
                        theme=args.tema, sound=not args.sem_som and args.frames is None)
    window.run(max_frames=args.frames)
    pg.quit()


if __name__ == "__main__":
    sys.exit(main())
