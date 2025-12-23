from plot.ci_ellipse_plot import plot_ci_ellipse

# =========================
# Setzungen (hier ändern!)
# =========================
n = 80
gamma = 0.95
h = 0.63

k = 9  # Anzahl der sichtbaren Prognoseintervalle im CI

pL, pR = plot_ci_ellipse(n=n, gamma=gamma, h_obs=h, k=k, show=True, outpath=None)

print("CI (aus Band-Schnitt):", (round(pL, 4), round(pR, 4)))
