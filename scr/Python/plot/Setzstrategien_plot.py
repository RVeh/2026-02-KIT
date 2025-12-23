# plot.py
from __future__ import annotations

from typing import Dict, List, Sequence, Optional

import matplotlib.pyplot as plt


def plot_sim_vs_exact(
    sim_by_runs: Dict[int, Dict[str, Dict[str, float]]],
    exact_floats: Sequence[float],
    labels: Sequence[str],
    save_path: Optional[str] = None,
    suptitle: str = "Simulation vs. exakte Werte für verschiedene Stichprobenumfänge",
):
    """
    sim_by_runs: {n_runs: {"A": {"p_hat":..., "CI_low":..., "CI_high":...}, "B":..., "U":...}}
    exact_floats: [P(A), P(B), P(U)]
    """
    n_runs_list = sorted(sim_by_runs.keys())

    fig, axes = plt.subplots(1, len(n_runs_list), figsize=(5 * len(n_runs_list), 5), sharey=True)
    if len(n_runs_list) == 1:
        axes = [axes]

    for ax in axes:
        ax.tick_params(axis="y", labelsize=14)

    for ax, n_runs in zip(axes, n_runs_list):
        sim = sim_by_runs[n_runs]
        sim_vals = [sim[k]["p_hat"] for k in ("A", "B", "U")]
        ci_lows = [sim[k]["CI_low"] for k in ("A", "B", "U")]
        ci_highs = [sim[k]["CI_high"] for k in ("A", "B", "U")]

        yerr = [
            [sim_vals[i] - ci_lows[i] for i in range(3)],
            [ci_highs[i] - sim_vals[i] for i in range(3)],
        ]

        ax.bar(
            range(3),
            sim_vals,
            yerr=yerr,
            align="center",
            alpha=0.7,
            capsize=8,
            label=f"Simulation (n={n_runs})",
            color="skyblue",
            width=0.25,
        )
        ax.scatter(range(3), list(exact_floats), color="red", zorder=5, label="Exakter Wert")

        ax.set_xticks(range(3))
        ax.set_xticklabels(list(labels), rotation=15, fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_title(f"n = {n_runs}")
        ax.legend(fontsize=12)

    fig.suptitle(suptitle, fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
