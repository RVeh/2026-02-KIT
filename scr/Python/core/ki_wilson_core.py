from math import sqrt
from statistics import NormalDist
import numpy as np

def z_value(gamma):
    alpha = 1 - gamma
    return NormalDist().inv_cdf(1 - alpha/2)

def wilson_ci(k, n, z):
    h = k / n
    denom = 1 + z**2 / n
    center = (h + z**2/(2*n)) / denom
    half = z * sqrt((h*(1-h)/n) + (z**2)/(4*n**2)) / denom
    return center - half, center + half

def simulate_intervals(n, p_true, gamma, m, seed):
    rng = np.random.default_rng(seed)
    z = z_value(gamma)
    X = rng.binomial(n, p_true, size=m)

    intervals = []
    cover = []
    for k in X:
        L, R = wilson_ci(int(k), n, z)
        intervals.append((L, R))
        cover.append(L <= p_true <= R)

    return np.array(intervals), np.array(cover), float(np.mean(cover))
