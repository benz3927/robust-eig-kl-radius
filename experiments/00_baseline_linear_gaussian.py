"""Baseline: NMC estimate of EIG vs closed form, plus robust EIG for a few radii.

Run from repo root:  python experiments/00_baseline_linear_gaussian.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eig import eig_closed_form, eig_samples_nmc, kl_worst_case_mean

SEED = 2026
xs = np.linspace(0.0, 3.0, 13)
eps_list = [0.0, 0.05, 0.2, 0.5]

exact = eig_closed_form(xs)
rows = []
for i, x in enumerate(xs):
    g = eig_samples_nmc(x, rng=SEED + i)
    row = [x, exact[i]] + [kl_worst_case_mean(g, e)[0] for e in eps_list]
    rows.append(row)
rows = np.array(rows)

os.makedirs("results", exist_ok=True)
header = "x,eig_exact," + ",".join(f"eig_eps_{e}" for e in eps_list)
np.savetxt("results/00_baseline.csv", rows, delimiter=",", header=header, comments="")

plt.figure(figsize=(6, 4))
plt.plot(xs, exact, "k-", lw=2, label="closed form")
for j, e in enumerate(eps_list):
    plt.plot(xs, rows[:, 2 + j], "o--", ms=3, label=f"NMC, eps = {e}")
plt.xlabel("design x")
plt.ylabel("(robust) EIG")
plt.legend()
plt.tight_layout()
plt.savefig("results/00_baseline.png", dpi=150)

err = np.max(np.abs(rows[:, 2] - exact))
print(f"max |NMC - closed form| at eps=0: {err:.4f}")
print("saved results/00_baseline.csv and results/00_baseline.png")
