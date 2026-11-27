"""Pong game logic, usable by a human (games/pong/play.py) and by learning agents.

No pygame here: the environment is pure Python, deterministic for a given seed, and fast enough
to train agents without a window.

    from games.pong import PongEnv
    env = PongEnv(seed=0)
    obs = env.reset()
    obs, reward, done, info = env.step(PongEnv.UP)

Player (agent or human) = right paddle. CPU = left paddle.

Actions (3):      0 = stay, 1 = up, 2 = down
Observation (6):  all in [-1, 1]
    0 ball x        (-1 left wall, +1 right wall)
    1 ball y        (-1 top, +1 bottom)
    2 ball vx       (divided by the maximum ball speed)
    3 ball vy       (divided by the maximum ball speed)
    4 player paddle centre y
    5 CPU paddle centre y
Reward:  +1 player scores, -1 CPU scores, +hit_reward when the player returns the ball.
Episode ends when one side reaches `points_to_win` or after `max_steps` steps.
One step = one frame at 60 FPS.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PongConfig:
    width: int = 1280
    height: int = 720
    paddle_w: int = 10
    paddle_h: int = 100
    paddle_margin: int = 10          # gap between paddle and side wall
    ball_size: int = 15
    wall: int = 2                    # thickness of top and bottom walls
    player_speed: float = 9.0
    cpu_speed: float = 6.0           # slower than the player, so the CPU can be beaten
    cpu_reaction: float = 0.50       # CPU reacts once the ball passes this fraction of the width
    ball_speed: float = 7.0
    ball_speed_max: float = 14.0     # must stay below paddle_w + ball_size (no tunnelling)
    ball_speedup: float = 1.06       # factor per paddle hit
    max_bounce_deg: float = 60.0     # angle when the ball hits the paddle edge
    points_to_win: int = 7
    max_steps: int = 30_000
    hit_reward: float = 0.1


class PongEnv:
    STAY, UP, DOWN = 0, 1, 2
    N_ACTIONS = 3
    OBS_SIZE = 6
    ACTION_NAMES = ("stay", "up", "down")

    def __init__(self, config: PongConfig | None = None, seed: int | None = None, **overrides):
        base = config or PongConfig()
        self.cfg = replace(base, **overrides)
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = random.Random(seed)
        c = self.cfg
        self.player_y = (c.height - c.paddle_h) / 2   # top of the paddle
        self.cpu_y = (c.height - c.paddle_h) / 2
        self.score_player = 0
        self.score_cpu = 0
        self.steps = 0
        self.hits = 0
        self.done = False
        self.last_event = None       # "player_hit", "cpu_hit", "wall", "point_player", "point_cpu"
        self._serve(direction=self.rng.choice((-1, 1)))
        return self.observation()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode finished: call reset()")
        if action not in (0, 1, 2):
            raise ValueError(f"action must be 0, 1 or 2, got {action!r}")
        c = self.cfg
        self.steps += 1
        self.last_event = None
        reward = 0.0

        # paddles
        if action == self.UP:
            self.player_y -= c.player_speed
        elif action == self.DOWN:
            self.player_y += c.player_speed
        self.player_y = self._clamp_paddle(self.player_y)
        self._move_cpu()

        # ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        top, bottom = c.wall, c.height - c.wall - c.ball_size
        if self.ball_y <= top:
            self.ball_y, self.ball_vy = top, abs(self.ball_vy)
            self.last_event = "wall"
        elif self.ball_y >= bottom:
            self.ball_y, self.ball_vy = bottom, -abs(self.ball_vy)
            self.last_event = "wall"

        if self.ball_vx > 0 and self._touches(self.player_x, self.player_y):
            self._bounce(self.player_y, direction=-1)
            self.ball_x = self.player_x - c.ball_size        # place it outside: no sticking
            self.last_event = "player_hit"
            reward += c.hit_reward
        elif self.ball_vx < 0 and self._touches(self.cpu_x, self.cpu_y):
            self._bounce(self.cpu_y, direction=+1)
            self.ball_x = self.cpu_x + c.paddle_w
            self.last_event = "cpu_hit"

        # points
        if self.ball_x + c.ball_size < 0:
            self.score_player += 1
            reward += 1.0
            self.last_event = "point_player"
            self._serve(direction=-1)        # serve towards whoever conceded
        elif self.ball_x > c.width:
            self.score_cpu += 1
            reward -= 1.0
            self.last_event = "point_cpu"
            self._serve(direction=+1)

        won = max(self.score_player, self.score_cpu) >= c.points_to_win
        self.done = won or self.steps >= c.max_steps
        info = {
            "score_player": self.score_player,
            "score_cpu": self.score_cpu,
            "hits": self.hits,
            "event": self.last_event,
            "truncated": self.done and not won,
        }
        return self.observation(), reward, self.done, info

    def observation(self):
        c = self.cfg
        vmax = c.ball_speed_max
        cx = self.ball_x + c.ball_size / 2
        cy = self.ball_y + c.ball_size / 2
        return [
            2 * cx / c.width - 1,
            2 * cy / c.height - 1,
            self.ball_vx / vmax,
            self.ball_vy / vmax,
            2 * (self.player_y + c.paddle_h / 2) / c.height - 1,
            2 * (self.cpu_y + c.paddle_h / 2) / c.height - 1,
        ]

    @property
    def winner(self):
        if self.score_player >= self.cfg.points_to_win:
            return "player"
        if self.score_cpu >= self.cfg.points_to_win:
            return "cpu"
        return None

    # ---------------------------------------------------- geometry helpers
    @property
    def player_x(self):
        return self.cfg.width - self.cfg.paddle_margin - self.cfg.paddle_w

    @property
    def cpu_x(self):
        return self.cfg.paddle_margin

    def _clamp_paddle(self, y):
        c = self.cfg
        return min(max(y, c.wall), c.height - c.wall - c.paddle_h)

    def _touches(self, px, py):
        c = self.cfg
        return (self.ball_x < px + c.paddle_w and self.ball_x + c.ball_size > px
                and self.ball_y < py + c.paddle_h and self.ball_y + c.ball_size > py)

    def _bounce(self, paddle_y, direction):
        """Angle depends on where the ball hits: centre = straight, edge = max_bounce_deg."""
        c = self.cfg
        offset = ((self.ball_y + c.ball_size / 2) - (paddle_y + c.paddle_h / 2)) / (c.paddle_h / 2)
        offset = max(-1.0, min(1.0, offset))
        angle = math.radians(c.max_bounce_deg) * offset
        speed = min(math.hypot(self.ball_vx, self.ball_vy) * c.ball_speedup, c.ball_speed_max)
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)
        self.hits += 1

    def _serve(self, direction):
        c = self.cfg
        self.ball_x = (c.width - c.ball_size) / 2
        self.ball_y = (c.height - c.ball_size) / 2
        angle = math.radians(self.rng.uniform(-30, 30))
        self.ball_vx = direction * c.ball_speed * math.cos(angle)
        self.ball_vy = c.ball_speed * math.sin(angle)

    def _move_cpu(self):
        """Simple opponent: follows the ball when it comes its way, otherwise drifts to the centre."""
        c = self.cfg
        centre = self.cpu_y + c.paddle_h / 2
        if self.ball_vx < 0 and self.ball_x < c.width * c.cpu_reaction:
            target = self.ball_y + c.ball_size / 2
        else:
            target = c.height / 2
        delta = target - centre
        if abs(delta) > c.paddle_h * 0.15:           # small dead zone: less jitter, beatable
            self.cpu_y += max(-c.cpu_speed, min(c.cpu_speed, delta))
        self.cpu_y = self._clamp_paddle(self.cpu_y)


def follow_ball_policy(obs, dead_zone=0.03):
    """Rule-based baseline: move the player paddle towards the ball. Useful as a sanity check
    and as the reference a trained agent should beat."""
    ball_y, paddle_y = obs[1], obs[4]
    if ball_y < paddle_y - dead_zone:
        return PongEnv.UP
    if ball_y > paddle_y + dead_zone:
        return PongEnv.DOWN
    return PongEnv.STAY
