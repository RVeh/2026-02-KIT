from __future__ import annotations

import matplotlib.pyplot as plt

from core.binom_exact_core import ExactSpec
from plot.binom_exact_plot import plot_exact_test, PlotCfg


def show_binom_exact(
    n: int = 80,
    p0: float = 0.5,
    alpha: float = 0.05,
    runs: int = 1000,
    seed: int | None = 42,
    side: str = "right",     # "left" | "right" | "two"
    nsigma: float = 5.0,
    prefer_scipy: bool = True,
):
    spec = ExactSpec(n=n, p0=p0, alpha=alpha, side=side)  # type: ignore[arg-type]
    cfg = PlotCfg(nsigma=nsigma)

    fig, ax, region, a_ex, a_hat = plot_exact_test(
        spec=spec,
        runs=runs,
        seed=seed,
        prefer_scipy=prefer_scipy,
        cfg=cfg,
    )

    plt.show()
    plt.close(fig)
    return region, a_ex, a_hat
