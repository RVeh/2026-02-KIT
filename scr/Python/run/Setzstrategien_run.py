# run.py
from __future__ import annotations

from fractions import Fraction

from core import exact_probabilities_fraction, simulate_many
from plot import plot_sim_vs_exact


def main():
    # --- Parameter (wie in deinem Original) ---
    p = [
        Fraction(3, 18), Fraction(5, 18), Fraction(4, 18),
        Fraction(3, 18), Fraction(2, 18), Fraction(1, 18),
    ]
    A = [3, 5, 4, 3, 2, 1]
    B = [3, 7, 4, 3, 1, 0]

    # In deinem Original sind das die Simulation-Laufzahlen pro Panel:
    n_runs_list = [50, 200]

    labels = ["A gewinnt", "B gewinnt", "Unentschieden"]

    # --- Exakt ---
    exact_vals = exact_probabilities_fraction(p, A, B)
    exact_floats = [float(v) for v in exact_vals]

    # --- Simulationen (für jeden n_runs) ---
    sim_by_runs = {}
    for n_runs in n_runs_list:
        sim = simulate_many([float(x) for x in p], A, B, n_runs=n_runs)
        sim_by_runs[n_runs] = sim

        # --- Tabellenausgabe wie vorher ---
        print(f"\n--- Ergebnisse für n = {n_runs} ---")
        for key, name, exact_val in zip(("A", "B", "U"), labels, exact_vals):
            phat = sim[key]["p_hat"]
            low = sim[key]["CI_low"]
            high = sim[key]["CI_high"]
            deviation = abs(phat - float(exact_val))
            print(
                f"{name:15s}: phat = {phat:.5f}, CI = ({low:.5f} – {high:.5f}), "
                f"Abweichung = {deviation:.5f}, exakt = {exact_val}"
            )

    # --- Plot ---
    plot_sim_vs_exact(
        sim_by_runs=sim_by_runs,
        exact_floats=exact_floats,
        labels=labels,
        save_path="sim_exakt_CI_n50-200-07.pdf",
    )


if __name__ == "__main__":
    main()
