from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np

Side = Literal["right", "left", "two"]


@dataclass(frozen=True)
class BinomTestSpec:
    n: int
    p0: float
    alpha: float
    side: Side = "right"


@dataclass(frozen=True)
class BinomCriticalRegion:
    k_left: int | None
    k_right: int | None


def _try_scipy_binom():
    try:
        from scipy.stats import binom  # type: ignore
        return binom
    except Exception:
        return None


def _binom_sf_scipy(k_minus_1: int, n: int, p: float) -> float:
    # P(X >= k) = sf(k-1)
    binom = _try_scipy_binom()
    if binom is None:
        raise ImportError("scipy ist nicht verfügbar.")
    return float(binom.sf(k_minus_1, n, p))


def _binom_cdf_scipy(k: int, n: int, p: float) -> float:
    binom = _try_scipy_binom()
    if binom is None:
        raise ImportError("scipy ist nicht verfügbar.")
    return float(binom.cdf(k, n, p))


def _pmf_recursion(n: int, p: float) -> np.ndarray:
    """
    Exakte Binomial-PMF via Rekursion (keine SciPy-Abhängigkeit).
    Stabil genug für typische Unterrichts-n (bis ein paar Tausend ok).
    """
    pmf = np.empty(n + 1, dtype=float)
    if p <= 0.0:
        pmf[:] = 0.0
        pmf[0] = 1.0
        return pmf
    if p >= 1.0:
        pmf[:] = 0.0
        pmf[n] = 1.0
        return pmf

    q = 1.0 - p
    pmf[0] = q ** n
    ratio = p / q
    for k in range(1, n + 1):
        pmf[k] = pmf[k - 1] * (n - k + 1) / k * ratio
    return pmf


def binom_sf_exact(k: int, n: int, p: float) -> float:
    """P(X >= k) exakt ohne SciPy."""
    pmf = _pmf_recursion(n, p)
    return float(pmf[k:].sum())


def binom_cdf_exact(k: int, n: int, p: float) -> float:
    """P(X <= k) exakt ohne SciPy."""
    pmf = _pmf_recursion(n, p)
    return float(pmf[: k + 1].sum())


def critical_region(spec: BinomTestSpec, prefer_scipy: bool = True) -> BinomCriticalRegion:
    """
    Kritischer Bereich für Bin(n,p0) unter H0.
    - right:  K = {k >= k_right}
    - left:   K = {k <= k_left}
    - two:    K = {k <= k_left oder k >= k_right} mit alpha/2 in den Enden
    """
    n, p0, alpha, side = spec.n, spec.p0, spec.alpha, spec.side

    use_scipy = prefer_scipy and (_try_scipy_binom() is not None)

    if side == "right":
        for k in range(0, n + 1):
            sf = _binom_sf_scipy(k - 1, n, p0) if use_scipy else binom_sf_exact(k, n, p0)
            if sf <= alpha:
                return BinomCriticalRegion(k_left=None, k_right=k)
        return BinomCriticalRegion(None, None)

    if side == "left":
        for k in range(0, n + 1):
            cdf = _binom_cdf_scipy(k, n, p0) if use_scipy else binom_cdf_exact(k, n, p0)
            if cdf <= alpha:
                k_left = k
            else:
                break
        return BinomCriticalRegion(k_left=k_left if "k_left" in locals() else None, k_right=None)

    # two-sided
    a2 = alpha / 2.0

    # left cutoff: largest k with CDF <= a2
    k_left = None
    for k in range(0, n + 1):
        cdf = _binom_cdf_scipy(k, n, p0) if use_scipy else binom_cdf_exact(k, n, p0)
        if cdf <= a2:
            k_left = k
        else:
            break

    # right cutoff: smallest k with SF <= a2
    k_right = None
    for k in range(0, n + 1):
        sf = _binom_sf_scipy(k - 1, n, p0) if use_scipy else binom_sf_exact(k, n, p0)
        if sf <= a2:
            k_right = k
            break

    return BinomCriticalRegion(k_left=k_left, k_right=k_right)


def simulate_binom(n: int, p: float, runs: int, seed: int | None = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(n, p, size=runs)


def alpha_hat_from_region(X: np.ndarray, region: BinomCriticalRegion, side: Side) -> float:
    if side == "right":
        if region.k_right is None:
            return 0.0
        return float(np.mean(X >= region.k_right))
    if side == "left":
        if region.k_left is None:
            return 0.0
        return float(np.mean(X <= region.k_left))
    # two-sided
    left = (X <= region.k_left) if region.k_left is not None else False
    right = (X >= region.k_right) if region.k_right is not None else False
    return float(np.mean(left | right))


def window_mu_sigma(n: int, p: float, nsigma: float = 5.0) -> tuple[int, int, float, float]:
    mu = n * p
    sigma = sqrt(n * p * (1.0 - p))
    if sigma <= 0:
        return 0, n, mu, sigma
    lo = max(0, int(np.floor(mu - nsigma * sigma)))
    hi = min(n, int(np.ceil(mu + nsigma * sigma)))
    return lo, hi, mu, sigma
