"""Exp 00 - Baseline: EIG and REIG on the linear-Gaussian model.

Three versions of REIG at each design x and radius eps:
  - exact:   closed form, no Monte Carlo error (reig_closed_form)
  - sampled: Algorithm 1 with nested-MC d_i (N1=500, N2=10, M=500)
  - joint:   the joint-ambiguity criterion of eq. (20), for comparison only
Run from repo root:  python experiments/00_baseline_linear_gaussian.py
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
from src.eig import (eig_closed_form, reig_closed_form, d_samples_nmc,
                     log_ratio_samples_nmc, reig_dual)

SEED = 2026
xs = np.linspace(0.0, 3.0, 13)
eps_list = [0.0, 0.05, 0.2, 0.5]
k = len(eps_list)

rows = []
for i, x in enumerate(xs):
    _, d_hat = d_samples_nmc(x, rng=SEED + i)
    g_joint = log_ratio_samples_nmc(x, rng=SEED + 1000 + i)
    row = [x, eig_closed_form(x)]
    row += [reig_closed_form(x, e) for e in eps_list]
    row += [reig_dual(d_hat, e)[0] for e in eps_list]
    row += [reig_dual(g_joint, e)[0] for e in eps_list]
    rows.append(row)
rows = np.array(rows)
EX, SA, JO = 2, 2 + k, 2 + 2 * k

header = ",".join(["x", "eig_exact"] + [f"reig_exact_eps_{e}" for e in eps_list]
                  + [f"reig_sampled_eps_{e}" for e in eps_list]
                  + [f"joint_eps_{e}" for e in eps_list])
np.savetxt(f"{RESULTS}/00_baseline.csv", rows, delimiter=",", header=header, comments="")

colors = plt.cm.viridis(np.linspace(0, 0.85, k))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax = axes[0]
for j, e in enumerate(eps_list):
    ax.plot(xs, rows[:, EX + j], "-", color=colors[j], lw=2, label=f"exact, eps = {e}")
    ax.plot(xs, rows[:, SA + j], "o", ms=4, color=colors[j])
ax.plot([], [], "o", color="gray", label="sampled (Algorithm 1)")
ax.set_title("REIG (KL ball on the prior): exact lines, sampled dots", fontsize=10)
ax = axes[1]
ax.plot(xs, rows[:, 1], "k-", lw=2, label="EIG")
for j, e in enumerate(eps_list[1:], start=1):
    ax.plot(xs, rows[:, JO + j], "o--", ms=3, color=colors[j], label=f"eps = {e}")
ax.set_title("Joint KL ball (eq. 20), for comparison only", fontsize=10)
for ax in axes:
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("design x"); ax.legend(fontsize=7)
axes[0].set_ylabel("information gain (nats)")
plt.tight_layout(); plt.savefig(f"{RESULTS}/00_baseline.png", dpi=150)

print(f"max |sampled EIG - exact| (eps=0):     {np.abs(rows[:, SA] - rows[:, 1]).max():.4f}")
print(f"max |sampled REIG - exact REIG|:       {np.abs(rows[:, SA:JO] - rows[:, EX:SA]).max():.4f}")
print(f"min exact REIG:                        {rows[:, EX:SA].min():.4f}")
print(f"min sampled REIG:                      {rows[:, SA:JO].min():.4f}")
print(f"min joint-ball value:                  {rows[:, JO:].min():.4f}")
