from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from core.ci_ellipse_core import band_h, z_value, invert_band_to_ci, prognose_schnitte


def plot_ci_ellipse(
    n: int,
    gamma: float,
    h_obs: float,
    k: int = 9,
    show: bool = True,
    outpath: Path | None = None,
):
    """
    Didaktische Darstellung der "Konfidenzellipse" (Prognoseband in (p,h)-Ebene).

    Elemente:
    - Band: h = p ± z * sqrt(p(1-p)/n) (grau gefüllt)
    - oberer Ast: blau, unterer Ast: grün
    - beobachtetes h: horizontale schwarze Linie
    - CI für p (aus Band-Schnitt): rote Strecke unten
    - k Prognoseintervalle (vertikale Schnitte) im CI: schwarze Linien
    - gleiche Skalierung (p- und h-Achse)
    """
    z = z_value(gamma)

    # Kurven fein genug für glatte Darstellung
    p = np.linspace(0.0, 1.0, 2000)
    lower, upper = band_h(p, n, z)

    # CI als Schnitt der Horizontalen h=h_obs mit dem Band
    pL, pR = invert_band_to_ci(h_obs=h_obs, n=n, gamma=gamma)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))  

    # 1) Band (grau gefüllt)
    ax.fill_between(
        p, lower, upper,
        color="lightgray",
        alpha=0.6,
        #label="Konfidenzbereich"
    )

    # 2) Bandgrenzen
    ax.plot(p, upper, color="blue", linewidth=1.5, label="obere Grenze")
    ax.plot(p, lower, color="green", linewidth=1.5, label="untere Grenze")

    # 3) Beobachtetes h (Realisation)
    ax.axhline(h_obs, color="black", linewidth=1.0, label="Trefferanteil h")

    # 4) Prognoseintervalle (vertikale Schnitte) im CI
    if np.isfinite(pL) and np.isfinite(pR):
        ps = prognose_schnitte(pL, pR, k)
        for pi in ps:
            lo, up = band_h(np.array([pi]), n, z)
            ax.vlines(pi, lo[0], up[0], color="black", linewidth=2.0)

        # 5) CI unten (rot) – wie in deinem Bild
        ax.hlines(
            y=0.0, xmin=pL, xmax=pR,
            color="red",
            linewidth=4.0,
            label="Wilson-KI bei h"
        )
        # 6) CI oben (rot) 
        ax.hlines(
            y=h_obs, xmin=pL, xmax=pR,
            color="red",
            linewidth=1.5,
        )

    # Optional: gestrichelte Lotlinien zu pL/pR (wie im Screenshot)
    ax.vlines(pL, 0.0, h_obs, color="black", linestyles="--", linewidth=1.2)
    ax.vlines(pR, 0.0, h_obs, color="black", linestyles="--", linewidth=1.2)

    # Achsen / Layout
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")  # <- gleiche Skalierung (wichtig!)

    ax.set_xlabel("p",fontsize=14)
    ax.set_ylabel("h",fontsize=14)

    ax.set_title(f"Konfidenzellipse bei $n={n}$, $h={h_obs:.3g}$, $\\gamma={gamma:.2f}$")

    ax.grid(True, linewidth=1.0, alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath)

    if show:
        plt.show()

    plt.close(fig)
    return pL, pR
