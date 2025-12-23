from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from core.binom_test_core import (
    BinomTestSpec, BinomCriticalRegion, Side,
    critical_region, simulate_binom, alpha_hat_from_region, window_mu_sigma
)


@dataclass(frozen=True)
class PlotConfig:
    nsigma: float = 5.0
    bar_width: float = 0.5
    title_fontsize: int = 14


def _k_label(spec: BinomTestSpec, region: BinomCriticalRegion) -> str:
    if spec.side == "right":
        return rf"$K=\{{k \geq {region.k_right}\}}$"
    if spec.side == "left":
        return rf"$K=\{{k \leq {region.k_left}\}}$"
    return rf"$K=\{{k \leq {region.k_left}\ \text{{ oder }}\ k \geq {region.k_right}\}}$"


def _is_in_region(k: int, region: BinomCriticalRegion, side: Side) -> bool:
    if side == "right":
        return region.k_right is not None and k >= region.k_right
    if side == "left":
        return region.k_left is not None and k <= region.k_left
    left = region.k_left is not None and k <= region.k_left
    right = region.k_right is not None and k >= region.k_right
    return left or right


def plot_simulation_vs_exact_region(
    spec: BinomTestSpec,
    runs: int = 1000,
    seed: int | None = 42,
    prefer_scipy: bool = True,
    cfg: PlotConfig = PlotConfig(),
):
    """
    Plot: Simulation (relative frequencies) + exact critical region (computed under H0).
    Window: mu ± nsigma*sigma (clamped to [0,n]).
    Ensures K-text stays inside the plot.
    """
    region = critical_region(spec, prefer_scipy=prefer_scipy)
    X = simulate_binom(spec.n, spec.p0, runs=runs, seed=seed)
    a_hat = alpha_hat_from_region(X, region, spec.side)

    lo, hi, mu, sigma = window_mu_sigma(spec.n, spec.p0, nsigma=cfg.nsigma)

    # Histogram on integer support in [lo, hi]
    vals, counts = np.unique(X, return_counts=True)
    mask = (vals >= lo) & (vals <= hi)
    vals = vals[mask]
    counts = counts[mask]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    # colors per bar (only red/blue)
    colors = ["red" if _is_in_region(int(k), region, spec.side) else "blue" for k in vals]

    ax.bar(
        vals,
        counts / runs,
        width=cfg.bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    # show cutoff(s)
    if spec.side in ("right", "two") and region.k_right is not None:
        ax.axvline(region.k_right - 0.5, linestyle="--", linewidth=1.6, color="red")
    if spec.side in ("left", "two") and region.k_left is not None:
        ax.axvline(region.k_left + 0.5, linestyle="--", linewidth=1.6, color="red")

    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_xlabel("realisierte Werte von $X$")
    ax.set_ylabel("relative Häufigkeit")

    # Title
    ax.set_title(
        f"{runs} Realisationen von $X\\sim \\mathrm{{Bin}}({spec.n},{spec.p0})$\n"
        f"Analytisch (exakt): $\\alpha={spec.alpha}$  |  Empirisch: $\\hat\\alpha={a_hat:.4f}$",
        fontsize=cfg.title_fontsize,
    )

    # Put K-label inside window (clamp)
    ymax = ax.get_ylim()[1]
    x_text = lo + 0.02 * (hi - lo + 1)
    y_text = 0.92 * ymax
    label = _k_label(spec, region)

    ax.text(x_text, y_text, label, color="red", va="top", ha="left")

    fig.tight_layout()
    return fig, ax, region, a_hat
