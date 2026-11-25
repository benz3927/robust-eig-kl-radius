"""Exp 08 - Certified design choice with a PAIRED comparison.

Exp 07D compared separate intervals for each sensor; they were too wide to ever
separate the sensors (true gap 0.153 nats). Here both sensors use the SAME prior
draws, outcome noise and inner samples (common random numbers), and we bound the
DIFFERENCE directly:
    g_i = PCE_sharp_i - NMC_broad_i      (lower-biased for EIG_sharp - EIG_broad)
    LCB = REIG(g, eps_stat(N))           (95% lower bound for the difference)
Call "sharp is better" if LCB > 0; symmetric test for "broad is better".
Wrong certified calls should stay below ~5%.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
import src.sensors as S
from src.eig import reig_dual, eps_stat

Ms, Ns, REPS = [20, 100, 500], [100, 400, 1600], 300
rows = []
for m in Ms:
    for N in Ns:
        right = wrong = 0
        for r in range(REPS):
            seed = 100_000 * m + 10 * N + r
            ls = S.d_samples_indep("sharp", n1=N, m=m, contrastive=True, rng=seed)
            us = S.d_samples_indep("sharp", n1=N, m=m, contrastive=False, rng=seed)
            lb = S.d_samples_indep("broad", n1=N, m=m, contrastive=True, rng=seed)
            ub = S.d_samples_indep("broad", n1=N, m=m, contrastive=False, rng=seed)
            e = eps_stat(N)
            if reig_dual(ls - ub, e)[0] > 0: right += 1
            elif reig_dual(lb - us, e)[0] > 0: wrong += 1
        rows.append([m, N, right / REPS, wrong / REPS])
rows = np.array(rows)
np.savetxt(f"{RESULTS}/08_paired_certified_choice.csv", rows, delimiter=",", comments="",
           header="M,N,frac_certified_sharp(correct),frac_certified_broad(wrong)")
for r in rows:
    print(f"M={int(r[0]):<4} N={int(r[1]):<5} correct certified: {r[2]:.3f}  wrong certified: {r[3]:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
for m, c in zip(Ms, ["C1", "C0", "C2"]):
    s = rows[rows[:, 0] == m]
    ax.plot(s[:, 1], s[:, 2], "o-", color=c, label=f"correct call, M={m}")
    ax.plot(s[:, 1], s[:, 3], "x--", color=c, label=f"wrong call, M={m}")
ax.axhline(0.05, color="k", ls=":", lw=1)
ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("N (prior samples per design)"); ax.set_ylabel("fraction of runs")
ax.set_title("Certified sharp-vs-broad choice, paired comparison", fontsize=10)
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{RESULTS}/08_paired_certified_choice.png", dpi=150)
