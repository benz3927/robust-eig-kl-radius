"""Exp 03 - The paper's two-blood-test example (Sec. 3), computed exactly.

Both tests have the same EIG at prior 0.5. For each radius eps we compute
  - REIG (affine approximation, what the paper optimises)
  - I_true (real worst-case EIG over the KL ball, by grid search)
for both tests, and see which test each criterion prefers.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
from src import binary_tests as bt

r_b = bt.calibrate_rb(0.5)
eps_grid = np.linspace(0, 0.3, 61)
rows = []
for e in eps_grid:
    rows.append([e] + [bt.reig(t, 0.5, e, r_b) for t in "AB"]
                    + [bt.true_worst(t, 0.5, e, r_b) for t in "AB"])
rows = np.array(rows)
np.savetxt(f"{RESULTS}/03_two_tests.csv", rows, delimiter=",", comments="",
           header="eps,reig_A,reig_B,true_worst_A,true_worst_B")

print(f"test B error rate r = {r_b:.4f};  EIG at prior 0.5 = {bt.eig('A', .5, r_b):.4f} nats (both)")
for e in [0.0, 0.01, 0.05, 0.1, 0.2]:
    i = np.argmin(abs(eps_grid - e)); r = rows[i]
    print(f"eps={e:<4}: REIG A={r[1]:.4f} B={r[2]:.4f} -> {'A' if r[1]>r[2] else 'B' if r[2]>r[1] else 'tie'}"
          f" | true worst A={r[3]:.4f} B={r[4]:.4f} -> {'A' if r[3]>r[4] else 'B' if r[4]>r[3] else 'tie'}")

pis = np.linspace(0.01, 0.99, 197)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(pis, [bt.eig("A", p, r_b) for p in pis], label="test A")
axes[0].plot(pis, [bt.eig("B", p, r_b) for p in pis], label="test B")
axes[0].axvline(0.5, color="gray", lw=0.5)
axes[0].set_xlabel("prior P(condition)"); axes[0].set_ylabel("EIG (nats)")
axes[0].set_title("EIG vs prior (paper Fig. 1)", fontsize=10); axes[0].legend()
axes[1].plot(eps_grid, rows[:, 1], "C0-", label="REIG, A")
axes[1].plot(eps_grid, rows[:, 2], "C1-", label="REIG, B")
axes[1].plot(eps_grid, rows[:, 3], "C0--", label="true worst, A")
axes[1].plot(eps_grid, rows[:, 4], "C1--", label="true worst, B")
axes[1].set_xlabel("eps"); axes[1].set_ylabel("nats")
axes[1].set_title("Robust criteria vs radius", fontsize=10); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RESULTS}/03_two_tests.png", dpi=150)
