# scr/Python/plot/wilson_viz.py
import matplotlib.pyplot as plt
import numpy as np

from core.wilson_core import simulate_intervals


def simulate_and_show(n, p_true, gamma, m, seed):
    intervals, cover, coverage_rate = simulate_intervals(
        n=n, p_true=p_true, gamma=gamma, m=m, seed=seed
    )

    y = np.arange(1, m + 1)
    fig, ax = plt.subplots(figsize=(4.2, 6.2))

    for i, (L, R) in enumerate(intervals):
        ax.hlines(
            y[i], L, R,
            linewidth=(2.2 if cover[i] else 3.0),
            linestyles=("-" if cover[i] else "--")
        )

    ax.axvline(p_true, linewidth=2.0)
    ax.set_ylim(0, m + 1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("p")
    ax.set_ylabel("Realisierung")
    ax.set_title(
        f"{m} Intervalle (Wilson), n={n}, γ={gamma:.2f}\n"
        f"Trefferquote ≈ {coverage_rate:.2f}  |  Seed {seed}"
    )

    fig.tight_layout()
    plt.show()
    plt.close(fig)

    return coverage_rate
