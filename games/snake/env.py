"""Snake game logic, usable by a human (games/snake/play.py) and by learning agents.

Pure Python (no pygame), deterministic for a given seed.

    from games.snake import SnakeEnv
    env = SnakeEnv(seed=0)
    obs = env.reset()
    obs, reward, done, info = env.step(SnakeEnv.RIGHT)

Actions (4):      0 = up, 1 = down, 2 = left, 3 = right
                  (pressing the opposite of the current direction is ignored: the snake keeps going)
Observation (8):  all in [-1, 1]
    0 food dx        (food column - head column) / (grid_w - 1)
    1 food dy        (food row - head row) / (grid_h - 1)
    2 danger up      +1 if the cell above the head is a wall or body, else -1
    3 danger down
    4 danger left
    5 danger right
    6 direction dx   -1 left, 0, +1 right
    7 direction dy   -1 up,   0, +1 down
Reward:  +1 food, -1 death (wall or own body), -step_penalty per step (default 0.01),
         and the episode is cut (truncated) after `max_idle` steps without eating.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SnakeConfig:
    grid_w: int = 12
    grid_h: int = 12
    start_length: int = 3
    step_penalty: float = 0.01
    max_idle: int = 200              # steps without food before truncation (avoids endless loops)
    max_steps: int = 5_000


class SnakeEnv:
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    N_ACTIONS = 4
    OBS_SIZE = 8
    ACTION_NAMES = ("up", "down", "left", "right")
    DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))          # (dx, dy) per action; y grows downwards

    def __init__(self, config: SnakeConfig | None = None, seed: int | None = None, **overrides):
        base = config or SnakeConfig()
        self.cfg = replace(base, **overrides)
        if self.cfg.grid_w < 5 or self.cfg.grid_h < 5:
            raise ValueError("grid must be at least 5x5")
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = random.Random(seed)
        c = self.cfg
        hx, hy = c.grid_w // 2, c.grid_h // 2
        self.direction = self.RIGHT
        self.body = deque((hx - i, hy) for i in range(c.start_length))   # body[0] = head
        self.cells = set(self.body)
        self.score = 0
        self.steps = 0
        self.idle = 0
        self.done = False
        self.last_event = None        # "eat", "wall", "self", "idle"
        self._place_food()
        return self.observation()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode finished: call reset()")
        if action not in (0, 1, 2, 3):
            raise ValueError(f"action must be 0..3, got {action!r}")
        c = self.cfg
        self.steps += 1
        self.idle += 1
        self.last_event = None
        if not self._is_reverse(action):
            self.direction = action
        dx, dy = self.DIRS[self.direction]
        hx, hy = self.body[0]
        new = (hx + dx, hy + dy)
        reward = -c.step_penalty

        tail = self.body[-1]
        grows = new == self.food
        hits_wall = not (0 <= new[0] < c.grid_w and 0 <= new[1] < c.grid_h)
        # moving into the cell the tail is leaving is allowed (the tail moves away)
        hits_self = new in self.cells and not (new == tail and not grows)
        if hits_wall or hits_self:
            self.done = True
            self.last_event = "wall" if hits_wall else "self"
            return self.observation(), -1.0, True, self._info(truncated=False)

        self.body.appendleft(new)
        self.cells.add(new)
        if grows:
            self.score += 1
            self.idle = 0
            reward = 1.0
            self.last_event = "eat"
            if len(self.body) == c.grid_w * c.grid_h:      # board full: perfect game
                self.done = True
                return self.observation(), reward, True, self._info(truncated=False)
            self._place_food()
        else:
            self.body.pop()
            if tail != new:
                self.cells.discard(tail)

        truncated = self.idle >= c.max_idle or self.steps >= c.max_steps
        if truncated:
            self.done = True
            self.last_event = self.last_event or "idle"
        return self.observation(), reward, self.done, self._info(truncated=truncated)

    def observation(self):
        c = self.cfg
        hx, hy = self.body[0]
        fx, fy = self.food
        danger = [1.0 if self._blocked(hx + dx, hy + dy) else -1.0 for dx, dy in self.DIRS]
        ddx, ddy = self.DIRS[self.direction]
        return [(fx - hx) / (c.grid_w - 1), (fy - hy) / (c.grid_h - 1), *danger, float(ddx), float(ddy)]

    @property
    def head(self):
        return self.body[0]

    @property
    def length(self):
        return len(self.body)

    # ------------------------------------------------------------ helpers
    def _info(self, truncated):
        return {"score": self.score, "length": len(self.body), "event": self.last_event,
                "truncated": truncated}

    def _is_reverse(self, action):
        (ax, ay), (dx, dy) = self.DIRS[action], self.DIRS[self.direction]
        return len(self.body) > 1 and ax == -dx and ay == -dy

    def _blocked(self, x, y):
        c = self.cfg
        if not (0 <= x < c.grid_w and 0 <= y < c.grid_h):
            return True
        return (x, y) in self.cells and (x, y) != self.body[-1]

    def _place_food(self):
        c = self.cfg
        free = [(x, y) for y in range(c.grid_h) for x in range(c.grid_w) if (x, y) not in self.cells]
        self.food = self.rng.choice(free)


def greedy_policy(obs):
    """Rule-based baseline: walk towards the food, avoiding immediate danger when possible."""
    fdx, fdy = obs[0], obs[1]
    danger = obs[2:6]
    prefs = []
    if abs(fdx) >= abs(fdy):
        prefs += [SnakeEnv.RIGHT if fdx > 0 else SnakeEnv.LEFT, SnakeEnv.DOWN if fdy > 0 else SnakeEnv.UP]
    else:
        prefs += [SnakeEnv.DOWN if fdy > 0 else SnakeEnv.UP, SnakeEnv.RIGHT if fdx > 0 else SnakeEnv.LEFT]
    prefs += [a for a in range(4) if a not in prefs]
    for a in prefs:
        if danger[a] < 0:
            return a
    return prefs[0]
