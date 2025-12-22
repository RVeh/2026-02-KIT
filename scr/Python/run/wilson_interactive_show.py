from core.wilson_core import simulate_wilson_intervals
from plot.wilson_plot import plot_intervals


# ============================================================
# Parameter (hier dürfen SuS/LuL experimentieren)
# ============================================================
n = 80
p_true = 0.60
gamma = 0.95
m = 100

seed = 7   # Empfehlung: feste Zahl für reproduzierbare Diskussion
# seed = None  # bewusst: jedes Mal neue Simulation


intervals, cover, rate = simulate_wilson_intervals(
    n=n,
    p_true=p_true,
    gamma=gamma,
    m=m,
    seed=seed
)

title = (
    f"{m} Intervalle (Wilson), n={n}, γ={gamma:.2f}\n"
    f"Trefferquote ≈ {rate:.2f}  |  Seed {seed}"
)

plot_intervals(
    intervals=intervals,
    cover=cover,
    p_true=p_true,
    title=title,
    show=True,
    outpath=None
)

print("Trefferquote:", round(rate, 3))
