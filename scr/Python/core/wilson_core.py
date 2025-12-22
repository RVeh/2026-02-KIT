from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np


def z_value(gamma: float) -> float:
    """Zweiseitiger z-Wert der Standardnormalverteilung."""
    alpha = 1.0 - gamma
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilson_ci(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson-Konfidenzintervall für Anteilsparameter p."""
    h = k / n
    denom = 1.0 + (z**2) / n
    center = (h + (z**2) / (2.0 * n)) / denom
    half = z * sqrt((h * (1.0 - h) / n) + (z**2) / (4.0 * n**2)) / denom
    return center - half, center + half


def simulate_wilson_intervals(
    n: int,
    p_true: float,
    gamma: float,
    m: int,
    seed: int | None = 1,
):
    """
    Simuliert m Wilson-Intervalle im Binomialmodell und prüft Überdeckung.

    Returns:
      intervals: np.ndarray (m,2)
      cover:     np.ndarray (m,) bool
      rate:      float
    """
    z = z_value(gamma)
    rng = np.random.default_rng(seed)
    X = rng.binomial(n, p_true, size=m)

    intervals = np.empty((m, 2), dtype=float)
    cover = np.empty(m, dtype=bool)

    for i, k in enumerate(X):
        L, R = wilson_ci(int(k), n, z)
        intervals[i, 0] = L
        intervals[i, 1] = R
        cover[i] = (L <= p_true <= R)

    rate = float(cover.mean())
    return intervals, cover, rate
