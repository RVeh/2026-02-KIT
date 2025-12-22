from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_intervals(
    intervals: np.ndarray,
    cover: np.ndarray,
    p_true: float,
    title: str,
    show: bool = True,
    outpath: Path | None = None,
):
    """
    Reine Darstellung: plotten (und optional speichern).
    Keine Simulation, keine Statistiklogik.
    """
    m = len(intervals)
    y = np.arange(1, m + 1)

    fig, ax = plt.subplots(figsize=(4.2, 6.2))

    for i, (L, R) in enumerate(intervals):
        if cover[i]:
            ax.hlines(y[i], L, R, linewidth=1.6, color="blue")
        else:
            ax.hlines(y[i], L, R, linewidth=1.6, color="red")

    ax.axvline(p_true, linewidth=1.5, color="gray")

    ax.set_ylim(0, m + 1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("p")
    ax.set_ylabel("Realisierung")
    ax.set_title(title)

    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath)

    if show:
        plt.show()

    plt.close(fig)
