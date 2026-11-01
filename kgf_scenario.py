"""Szenario: Punktwolken in der Ebene (in die Länge gezogen und gedreht) und Normalwolken in d Dimensionen für die LP-Nachbildung."""

import math
from dataclasses import dataclass

import numpy as np

# Handinstanz (von Hand gerechnet, PLAN-02): Würfel 1, Raute 2, Kugel sqrt 2, Ellipse achsenparallel 4, Ellipse allgemein 2,5
HAND_POINTS = np.array([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [-0.5, 0.5]])


@dataclass(frozen=True, eq=False)
class Cloud:
    points: np.ndarray   # (n, 2), um den Mittelwert verschoben

    @property
    def n(self) -> int:
        return len(self.points)


def rotation(angle_deg: float) -> np.ndarray:
    th = math.radians(angle_deg)
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def make_cloud(n: int, stretch: float, angle_deg: float, seed: int) -> Cloud:
    """Normalverteilte Punkte, entlang der Längsachse um `stretch` breiter als quer dazu, die Längsachse um `angle_deg` gegen die x-Achse gedreht, Mittelwert null.

    Deterministisch aus (n, stretch, angle_deg, seed)."""
    if n < 2:
        raise ValueError("mindestens zwei Punkte")
    if stretch < 1.0:
        raise ValueError("Streckung muss mindestens 1 sein")
    z = np.random.default_rng(seed).normal(0.0, 1.0, (n, 2)) * np.array([stretch, 1.0])
    pts = z @ rotation(angle_deg).T
    return Cloud(pts - pts.mean(axis=0))


def make_gaussian(n: int, d: int, seed: int) -> np.ndarray:
    """Normalwolke in d Dimensionen (für die LP-Nachbildung der Kugel)."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, d))
