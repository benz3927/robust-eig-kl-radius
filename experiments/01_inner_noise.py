"""Exp 01 - How inner-estimator noise biases sampled REIG.

For designs x in {0.5, 2.0}, compare sampled REIG (Algorithm 1) with the exact
closed-form REIG while varying
    N2 = outcomes per theta_i    in {1, 5, 25}
    M  = prior samples in p_hat  in {10, 100, 1000}
N1 = 500, 10 repetitions per setting. Reports mean error (bias) and its spread.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
from src.eig import reig_closed_form, d_samples_nmc, reig_dual

xs, n2s, ms = [0.5, 2.0], [1, 5, 25], [10, 100, 1000]
eps_list = [0.0, 0.05, 0.2, 0.5]
REPS = 10

out = []
for x in xs:
    exact = {e: reig_closed_form(x, e) for e in eps_list}
    for n2 in n2s:
        for m in ms:
            errs = np.zeros((REPS, len(eps_list)))
            negs = 0
            for r in range(REPS):
                _, d = d_samples_nmc(x, n1=500, n2=n2, m=m, rng=10_000 * r + 100 * n2 + m)
                vals = [reig_dual(d, e)[0] for e in eps_list]
                errs[r] = np.array(vals) - np.array([exact[e] for e in eps_list])
                negs += int(min(vals) < 0)
            for j, e in enumerate(eps_list):
                out.append([x, n2, m, e, exact[e], errs[:, j].mean(), errs[:, j].std(), negs / REPS])
out = np.array(out)
np.savetxt(f"{RESULTS}/01_inner_noise.csv", out, delimiter=",", comments="",
           header="x,n2,m,eps,reig_exact,bias,sd,frac_runs_with_negative")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, x in zip(axes, xs):
    for n2, ls in zip(n2s, [":", "--", "-"]):
        for m, c in zip(ms, ["tab:red", "tab:orange", "tab:green"]):
            s = out[(out[:, 0] == x) & (out[:, 1] == n2) & (out[:, 2] == m)]
            ax.plot(s[:, 3], s[:, 5], ls, color=c, marker="o", ms=3,
                    label=f"N2={n2}, M={m}" if x == xs[0] else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"design x = {x}", fontsize=10); ax.set_xlabel("eps")
axes[0].set_ylabel("mean(sampled REIG - exact REIG)")
axes[0].legend(fontsize=6, ncol=3)
plt.tight_layout(); plt.savefig(f"{RESULTS}/01_inner_noise.png", dpi=150)

for row in out[out[:, 3] == 0.5]:
    print(f"x={row[0]} N2={int(row[1]):>2} M={int(row[2]):>4} eps=0.5: bias={row[5]:+.3f} sd={row[6]:.3f} neg={row[7]:.1f}")
