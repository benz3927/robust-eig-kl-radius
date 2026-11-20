"""Exp 04 - A first eps-calibration rule (exploratory).

Idea from the paper's Sec. 8: plain nested MC (NMC) over-estimates EIG and the
contrastive estimator (include theta_i, like ACE) under-estimates it. The paper
applies REIG (min) to the over-estimator and REIG-max (eq. 19) to the
under-estimator. Both move toward the truth as eps grows.

Rule tested here: pick eps* where the two curves meet, and use that common
value as the EIG estimate. Compared with plain NMC, contrastive, and their
average, with a small inner budget (M = 10) where the biases are large.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
from src.eig import eig_closed_form, d_samples_nmc, reig_dual, reig_max_dual, crossing_estimate

xs = np.linspace(0.25, 3.0, 12)
M, N1, N2, REPS = 10, 500, 10, 10




rows = []
for i, x in enumerate(xs):
    truth = eig_closed_form(x)
    for r in range(REPS):
        seed = 1000 * i + r
        _, d_up = d_samples_nmc(x, n1=N1, n2=N2, m=M, rng=seed)
        _, d_lo = d_samples_nmc(x, n1=N1, n2=N2, m=M, contrastive=True, rng=seed)
        e_star, est = crossing_estimate(d_up, d_lo)
        rows.append([x, r, truth, d_up.mean(), d_lo.mean(),
                     0.5 * (d_up.mean() + d_lo.mean()), est, e_star])
rows = np.array(rows)
np.savetxt(f"{RESULTS}/04_eps_calibration.csv", rows, delimiter=",", comments="",
           header="x,rep,eig_true,nmc_upper,contrastive_lower,average,crossing_estimate,eps_star")

names = ["NMC (upper)", "contrastive (lower)", "average", "REIG crossing"]
for j, n in enumerate(names):
    err = rows[:, 3 + j] - rows[:, 2]
    print(f"{n:<20} mean error {err.mean():+.4f}   RMSE {np.sqrt((err**2).mean()):.4f}")
print(f"eps* range: {rows[:, 7].min():.3f} - {rows[:, 7].max():.3f}")

g = lambda col: np.array([rows[rows[:, 0] == x, col].mean() for x in xs])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(xs, g(2), "k-", lw=2, label="true EIG")
for j, (n, st) in enumerate(zip(names, ["v--", "^--", "s:", "o-"])):
    axes[0].plot(xs, g(3 + j), st, ms=4, label=n)
axes[0].set_xlabel("design x"); axes[0].set_ylabel("EIG (nats)"); axes[0].legend(fontsize=7)
axes[0].set_title(f"Estimates with small inner budget M={M}", fontsize=10)
axes[1].plot(xs, g(7), "o-")
axes[1].set_xlabel("design x"); axes[1].set_ylabel("eps* (mean over reps)")
axes[1].set_title("Radius chosen by the crossing rule", fontsize=10)
plt.tight_layout(); plt.savefig(f"{RESULTS}/04_eps_calibration.png", dpi=150)
