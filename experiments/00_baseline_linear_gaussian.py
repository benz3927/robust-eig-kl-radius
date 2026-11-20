"""Baseline: EIG and REIG on the linear-Gaussian model.

Checks:
  1. NMC estimate of EIG vs closed form.
  2. REIG from Algorithm 1 (sampled d_i) vs REIG from exact d(theta_i).
  3. Comparison with the joint-ambiguity version (eq. 20), which is what the
     first version of this repo computed by mistake and which can go negative.

Run from repo root:  python experiments/00_baseline_linear_gaussian.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eig import (eig_closed_form, d_closed_form, d_samples_nmc,
                     log_ratio_samples_nmc, reig_dual)

SEED = 2026
xs = np.linspace(0.0, 3.0, 13)
eps_list = [0.0, 0.05, 0.2, 0.5]

rows = []
for i, x in enumerate(xs):
    theta, d_hat = d_samples_nmc(x, rng=SEED + i)
    d_exact = d_closed_form(theta, x)
    g_joint = log_ratio_samples_nmc(x, rng=SEED + 1000 + i)
    row = [x, eig_closed_form(x)]
    row += [reig_dual(d_hat, e)[0] for e in eps_list]
    row += [reig_dual(d_exact, e)[0] for e in eps_list]
    row += [reig_dual(g_joint, e)[0] for e in eps_list]
    rows.append(row)
rows = np.array(rows)
k = len(eps_list)

os.makedirs("results", exist_ok=True)
header = ",".join(["x", "eig_exact"]
                  + [f"reig_nmc_eps_{e}" for e in eps_list]
                  + [f"reig_exactd_eps_{e}" for e in eps_list]
                  + [f"joint_eps_{e}" for e in eps_list])
np.savetxt("results/00_baseline.csv", rows, delimiter=",", header=header, comments="")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
colors = plt.cm.viridis(np.linspace(0, 0.85, k))
for ax, off, title in [(axes[0], 2, "REIG (Go & Isaac, KL ball on prior)"),
                       (axes[1], 2 + 2 * k, "Joint KL ball (eq. 20), for comparison")]:
    ax.plot(xs, rows[:, 1], "k-", lw=2, label="EIG, closed form")
    for j, e in enumerate(eps_list):
        ax.plot(xs, rows[:, off + j], "o--", ms=3, color=colors[j], label=f"eps = {e}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("design x")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("information gain (nats)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/00_baseline.png", dpi=150)

print(f"max |NMC EIG - closed form|:            {np.max(np.abs(rows[:, 2] - rows[:, 1])):.4f}")
print(f"max |REIG nmc - REIG exact d| (eps>0):  "
      f"{np.max(np.abs(rows[:, 3:2 + k] - rows[:, 3 + k:2 + 2 * k])):.4f}")
print(f"min REIG (prior ball), any eps:         {rows[:, 2:2 + k].min():.4f}")
print(f"min joint-ball value, any eps:          {rows[:, 2 + 2 * k:].min():.4f}")
print("saved results/00_baseline.csv and results/00_baseline.png")
