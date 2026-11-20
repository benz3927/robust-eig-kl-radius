"""Exp 06 - Does eps help pick the right sensor when EIG is estimated by sampling?

Same sensor model as Exp 05. Under the reference prior the sharp sensor is the
right choice (EIG 2.096 vs 1.943). With a small inner budget M, nested MC is
biased, and the bias differs between sensors. For each budget and 30 seeds we
pick the sensor with the highest estimate, using:
  - NMC (upper-biased), contrastive (lower-biased), their average
  - the Exp 04 crossing rule (REIG of NMC meets REIG-max of contrastive)
  - plain REIG of NMC at fixed eps = 0.05 and 0.2
and report how often each picks the sharp sensor, and the mean error of the estimates.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
import src.sensors as S
from src.eig import reig_dual, reig_max_dual

names = list(S.SENSORS)
truth = {k: S.eig(k) for k in names}
Ms, REPS, N1, N2 = [5, 20, 100, 500], 30, 300, 5


def crossing(d_up, d_lo, hi=2.0, iters=40):
    f = lambda e: reig_dual(d_up, e)[0] - reig_max_dual(d_lo, e)[0]
    if f(0.0) <= 0:
        return 0.0, 0.5 * (d_up.mean() + d_lo.mean())
    if f(hi) > 0:
        return hi, 0.5 * (reig_dual(d_up, hi)[0] + reig_max_dual(d_lo, hi)[0])
    lo_e, hi_e = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo_e + hi_e)
        lo_e, hi_e = (mid, hi_e) if f(mid) > 0 else (lo_e, mid)
    e = 0.5 * (lo_e + hi_e)
    return e, 0.5 * (reig_dual(d_up, e)[0] + reig_max_dual(d_lo, e)[0])


methods = ["NMC", "contrastive", "average", "crossing", "REIG eps=0.05", "REIG eps=0.2"]
rows = []
for m in Ms:
    for r in range(REPS):
        est = {}
        for j, k in enumerate(names):
            seed = 100_000 * m + 1000 * r + j
            du = S.d_samples(k, n1=N1, n2=N2, m=m, rng=seed)
            dl = S.d_samples(k, n1=N1, n2=N2, m=m, contrastive=True, rng=seed)
            est[k] = [du.mean(), dl.mean(), 0.5 * (du.mean() + dl.mean()), crossing(du, dl)[1],
                      reig_dual(du, 0.05)[0], reig_dual(du, 0.2)[0]]
        for i, meth in enumerate(methods):
            s, b = est["sharp"][i], est["broad"][i]
            rows.append([m, r, i, s, b, float(s > b)])
rows = np.array(rows)
np.savetxt(f"{RESULTS}/06_noisy_design_choice.csv", rows, delimiter=",", comments="",
           header="M,rep,method_index,est_sharp,est_broad,picked_sharp")

print(f"truth: sharp={truth['sharp']:.3f} broad={truth['broad']:.3f}  (methods: {methods})")
summary = np.zeros((len(Ms), len(methods)))
for a, m in enumerate(Ms):
    line = []
    for i, meth in enumerate(methods):
        s = rows[(rows[:, 0] == m) & (rows[:, 2] == i)]
        summary[a, i] = s[:, 5].mean()
        line.append(f"{meth}: {s[:, 5].mean():.2f} (err s {np.mean(s[:, 3] - truth['sharp']):+.2f}, "
                    f"b {np.mean(s[:, 4] - truth['broad']):+.2f})")
    print(f"M={m:<4}" + " | ".join(line))

fig, ax = plt.subplots(figsize=(7, 4))
for i, (meth, st) in enumerate(zip(methods, ["v--", "^--", "s:", "o-", "x-.", "d-."])):
    ax.plot(Ms, summary[:, i], st, label=meth)
ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("inner samples M (log scale)"); ax.set_ylabel("fraction of runs picking sharp (correct)")
ax.set_title("Picking the right sensor from noisy EIG estimates", fontsize=10)
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{RESULTS}/06_noisy_design_choice.png", dpi=150)
