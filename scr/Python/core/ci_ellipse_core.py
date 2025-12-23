from __future__ import annotations

import numpy as np
from statistics import NormalDist


def z_value(gamma: float) -> float:
    alpha = 1.0 - gamma
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def band_h(p: np.ndarray, n: int, z: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Prognoseband für H (Stichprobenanteil) als Funktion von p:
        h_lower(p) = p - z * sqrt(p(1-p)/n)
        h_upper(p) = p + z * sqrt(p(1-p)/n)
    """
    p = np.asarray(p, dtype=float)
    se = np.sqrt(np.clip(p * (1.0 - p) / n, 0.0, None))
    lower = p - z * se
    upper = p + z * se
    return lower, upper


def invert_band_to_ci(h_obs: float, n: int, gamma: float, grid: int = 20001) -> tuple[float, float]:
    """
    Invertiert das Band h = p ± z*sqrt(p(1-p)/n) zu einem CI für p
    mittels feiner Gitter-Suche (robust, für Visualisierung ausreichend).

    Liefert (p_left, p_right) so dass h_obs im Band liegt:
        lower(p) <= h_obs <= upper(p)
    """
    z = z_value(gamma)
    p = np.linspace(0.0, 1.0, grid)
    lower, upper = band_h(p, n, z)

    inside = (lower <= h_obs) & (h_obs <= upper)
    if not np.any(inside):
        # kann bei extremen h_obs und sehr kleinem n passieren; dann clampen wir sinnvoll
        return float("nan"), float("nan")

    idx = np.where(inside)[0]
    p_left = p[idx[0]]
    p_right = p[idx[-1]]
    return float(p_left), float(p_right)


def prognose_schnitte(p_left: float, p_right: float, k: int) -> np.ndarray:
    """
    Erzeugt k p-Werte innerhalb [p_left, p_right] für sichtbare Prognoseintervalle.
    """
    if k <= 0 or not np.isfinite(p_left) or not np.isfinite(p_right):
        return np.array([], dtype=float)
    if p_right < p_left:
        p_left, p_right = p_right, p_left
    return np.linspace(p_left, p_right, k)
