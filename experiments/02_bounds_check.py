"""Exp 02 - Check the paper's bounds (eq. 15-16) in the linear-Gaussian model.

Theory:   EIG(p) >= REIG_eps >= I_true_eps,   REIG_eps - I_true_eps <= eps.
We can't compute I_true exactly (min over ALL priors in the ball), but the
worst case over Gaussian priors G_eps is an upper bound on it. So we check
    EIG >= REIG,   and   REIG - G_eps <= eps   (a necessary condition)
and plot how far REIG sits above the Gaussian worst case as eps grows.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
from src.eig import eig_closed_form, reig_closed_form, eig_gaussian_worst_case

xs = [0.5, 1.0, 2.0, 3.0]
eps_grid = np.linspace(0, 1.0, 41)
rows = []
for x in xs:
    for e in eps_grid:
        rows.append([x, e, eig_closed_form(x), reig_closed_form(x, e), eig_gaussian_worst_case(x, e)])
rows = np.array(rows)
np.savetxt(f"{RESULTS}/02_bounds_check.csv", rows, delimiter=",", comments="",
           header="x,eps,eig,reig,gaussian_worst_case")

ok1 = np.all(rows[:, 2] >= rows[:, 3] - 1e-10)
ok2 = np.all(rows[:, 3] - rows[:, 4] <= rows[:, 1] + 1e-10)
print(f"EIG >= REIG everywhere:               {ok1}")
print(f"REIG - Gaussian worst <= eps:          {ok2}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for x, c in zip(xs, plt.cm.plasma(np.linspace(0, 0.8, len(xs)))):
    s = rows[rows[:, 0] == x]
    axes[0].plot(s[:, 1], s[:, 3], "-", color=c, label=f"REIG, x={x}")
    axes[0].plot(s[:, 1], s[:, 4], "--", color=c, label=f"Gaussian worst, x={x}")
    axes[1].plot(s[:, 1], s[:, 3] - s[:, 4], color=c, label=f"x={x}")
    print(f"x={x}: REIG - Gaussian worst at eps=0.1: {np.interp(0.1, s[:,1], s[:,3]-s[:,4]):.4f}, "
          f"at eps=1.0: {s[-1,3]-s[-1,4]:.4f}")
axes[1].plot(eps_grid, eps_grid, "k:", label="eps (eq. 16 limit)")
axes[0].set_xlabel("eps"); axes[0].set_ylabel("nats"); axes[0].legend(fontsize=6)
axes[0].set_title("REIG vs worst case over Gaussian priors", fontsize=10)
axes[1].set_xlabel("eps"); axes[1].set_ylabel("REIG - Gaussian worst case")
axes[1].set_title("Gap stays under eps but grows with it", fontsize=10); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{RESULTS}/02_bounds_check.png", dpi=150)
