"""Shared retro look for the games: white on black, blocky pixel digits, square beeps.

Our own 1970s-arcade style (no assets from any existing game): everything is drawn with
rectangles and the sounds are generated square waves.
"""
from __future__ import annotations

import pygame as pg

BLACK = (0, 0, 0)
WHITE = (236, 236, 236)
GREY = (120, 120, 120)
DIM = (0, 0, 0, 170)

# 3x5 pixel digits ("1" = lit block)
_DIGITS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def draw_number(screen, value, center_x, top, block=14, color=WHITE):
    """Draw an integer with blocky 3x5 digits, centred on center_x."""
    text = str(value)
    gap = block
    width = len(text) * 3 * block + (len(text) - 1) * gap
    x = center_x - width // 2
    for ch in text:
        for r, row in enumerate(_DIGITS[ch]):
            for col, bit in enumerate(row):
                if bit == "1":
                    pg.draw.rect(screen, color, (x + col * block, top + r * block, block, block))
        x += 3 * block + gap


def draw_dashed_vline(screen, x, height, dash=18, gap=14, width=6, color=WHITE):
    y = 0
    while y < height:
        pg.draw.rect(screen, color, (x - width // 2, y, width, dash))
        y += dash + gap


class Fonts:
    def __init__(self):
        pg.font.init()
        self.big = pg.font.Font(None, 80)
        self.small = pg.font.Font(None, 30)

    def text(self, screen, s, font, center, color=WHITE):
        surf = font.render(s, False, color)        # no anti-aliasing: crisp, pixel look
        screen.blit(surf, surf.get_rect(center=center))


def overlay(screen, fonts, title, line1=None, line2=None):
    """Dark veil + bordered panel with up to three lines of text."""
    w, h = screen.get_size()
    veil = pg.Surface((w, h), pg.SRCALPHA)
    veil.fill(DIM)
    screen.blit(veil, (0, 0))
    panel = pg.Rect(0, 0, min(780, w - 40), 210 if line2 else 160)
    panel.center = (w // 2, h // 2)
    pg.draw.rect(screen, BLACK, panel)
    pg.draw.rect(screen, WHITE, panel, width=4)
    fonts.text(screen, title.upper(), fonts.big, (w // 2, h // 2 - 50))
    if line1:
        fonts.text(screen, line1, fonts.small, (w // 2, h // 2 + 15))
    if line2:
        fonts.text(screen, line2, fonts.small, (w // 2, h // 2 + 55))


class Beeper:
    """Square-wave beeps. Silent (no error) when there is no audio device."""

    def __init__(self, enabled=True):
        self.sounds = {}
        if not enabled:
            return
        try:
            import numpy as np
            pg.mixer.pre_init(22050, -16, 1, 256)
            pg.mixer.init()
            rate = pg.mixer.get_init()[0]
            channels = pg.mixer.get_init()[2]
            for name, (freq, ms) in {"hit": (480, 40), "wall": (240, 30), "point": (120, 260),
                                      "eat": (660, 50), "dead": (90, 400)}.items():
                n = int(rate * ms / 1000)
                wave = np.where((np.arange(n) * freq * 2 // rate) % 2 == 0, 3000, -3000).astype(np.int16)
                if channels > 1:
                    wave = np.repeat(wave[:, None], channels, axis=1)
                self.sounds[name] = pg.sndarray.make_sound(np.ascontiguousarray(wave))
        except Exception:
            self.sounds = {}

    def play(self, name):
        snd = self.sounds.get(name)
        if snd is not None:
            snd.play()
