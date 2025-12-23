# core.py
from __future__ import annotations

import math
import random
from functools import lru_cache
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)

    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    halfwidth = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n))
    return (center - halfwidth, center + halfwidth)


def simulate_game(p: Sequence[float], A: Sequence[int], B: Sequence[int]) -> str:
    """
    Simulates one game.
    Returns: 'A' if A finishes first, 'B' if B finishes first, 'U' if tie.
    """
    m = len(p)
    chips_A = list(A)
    chips_B = list(B)

    def fertig(chips: List[int]) -> bool:
        return all(c == 0 for c in chips)

    while True:
        r = random.random()
        s = 0.0
        feld = 0
        for j in range(m):
            s += p[j]
            if r < s:
                feld = j
                break

        if chips_A[feld] > 0:
            chips_A[feld] -= 1
        if chips_B[feld] > 0:
            chips_B[feld] -= 1

        a_done = fertig(chips_A)
        b_done = fertig(chips_B)

        if a_done and b_done:
            return "U"
        if a_done:
            return "A"
        if b_done:
            return "B"


def simulate_many(
    p: Sequence[float],
    A: Sequence[int],
    B: Sequence[int],
    n_runs: int = 10_000,
    z: float = 1.96,
) -> Dict[str, Dict[str, float]]:
    """
    Runs many simulations and returns estimates + Wilson CIs for outcomes A/B/U.
    """
    counts = {"A": 0, "B": 0, "U": 0}
    for _ in range(n_runs):
        counts[simulate_game(p, A, B)] += 1

    results: Dict[str, Dict[str, float]] = {}
    for key in ("A", "B", "U"):
        phat = counts[key] / n_runs
        low, high = wilson_interval(counts[key], n_runs, z=z)
        results[key] = {"p_hat": phat, "CI_low": low, "CI_high": high}
    return results


def exact_probabilities_fraction(
    p: Sequence[Fraction],
    A: Sequence[int],
    B: Sequence[int],
) -> Tuple[Fraction, Fraction, Fraction]:
    """
    Exact probabilities using Fractions + memoization.
    Returns (P(A wins), P(B wins), P(tie)).
    """
    p = tuple(Fraction(pi) for pi in p)
    A = tuple(A)
    B = tuple(B)

    @lru_cache(None)
    def P_A(V: Tuple[int, ...], W: Tuple[int, ...]) -> Fraction:
        sum_V, sum_W = sum(V), sum(W)
        if sum_V == 0 and sum_W == 0:
            return Fraction(0, 1)   # tie terminal -> contributes 0 to P(A)
        if sum_V == 0:
            return Fraction(1, 1)   # A finished, B not -> A win
        if sum_W == 0:
            return Fraction(0, 1)   # B finished, A not -> A loses

        # Only fields where at least one player still has chips contribute to selection
        s = sum(pj for pj, vj, wj in zip(p, V, W) if (vj or wj))
        if s == 0:
            return Fraction(0, 1)

        prob = Fraction(0, 1)
        for j, pj in enumerate(p):
            if V[j] or W[j]:
                Vn = V[:j] + (max(0, V[j] - 1),) + V[j + 1 :]
                Wn = W[:j] + (max(0, W[j] - 1),) + W[j + 1 :]
                prob += (pj / s) * P_A(Vn, Wn)
        return prob

    PA = P_A(A, B)
    PB = P_A(B, A)
    PU = Fraction(1, 1) - PA - PB
    return PA, PB, PU
