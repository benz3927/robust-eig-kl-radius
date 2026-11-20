"""Exp 05 - A design problem where the robust choice flips with eps.

Sensor model (src/sensors.py): theta ~ N(0,1), y ~ N(tanh((theta - c)/w), 0.05^2).
  sharp: w = 0.5  (very informative near 0, saturates in the tails)
  broad: w = 2.5  (less sharp, informative over a wider range)
All values are on a fine grid (no Monte Carlo noise).

For each eps we compute, per sensor:
  - REIG (Go & Isaac, eq. 18)
  - worst EIG over Gaussian priors in the KL ball (mean/scale misspecified)
  - real EIG at the worst prior REIG itself finds (an exponential tilt of p)
The last two are feasible priors, so min(both) is an upper bound on the true
worst case I_true. The paper's theory says REIG >= I_true and REIG - I_true <= eps.

Then a decision view: if the real prior is anywhere within KL kappa of p, what is
the worst-case regret of picking each sensor?
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
import src.sensors as S

names = list(S.SENSORS)
eps_grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

rows, priors_by_eps = [], {}
for e in eps_grid:
    ball = S.gaussian_ball(e, n_angle=24, n_rad=1) if e > 0 else [(0.0, 1.0)]
    tilts = [S.reig_tilt_prior(k, e)[0] for k in names] if e > 0 else [S.prior_weights()]
    cands = [S.prior_weights(mu, tau) for mu, tau in ball] + tilts
    eigs = np.array([[S.eig_weights(k, q) for k in names] for q in cands])  # (n_priors, 2)
    priors_by_eps[e] = eigs
    row = [e]
    for j, k in enumerate(names):
        g_worst = eigs[:len(ball), j].min()
        t_worst = eigs[len(ball):, j].min()
        row += [S.reig(k, e), g_worst, t_worst, min(g_worst, t_worst)]
    rows.append(row)
rows = np.array(rows)
cols = ["eps"] + [f"{k}_{c}" for k in names for c in ["reig", "gauss_worst", "tilt_worst", "worst_bound"]]
np.savetxt(f"{RESULTS}/05_competing_designs.csv", rows, delimiter=",", comments="", header=",".join(cols))

# worst-case regret of each fixed choice, over all candidate priors at radius kappa
regret = []
for e in eps_grid:
    E = priors_by_eps[e]
    best = E.max(axis=1)
    regret.append([e] + [(best - E[:, j]).max() for j in range(len(names))])
regret = np.array(regret)
np.savetxt(f"{RESULTS}/05_regret.csv", regret, delimiter=",", comments="",
           header="kappa," + ",".join(f"worst_regret_{k}" for k in names))

print(f"plain EIG: " + ", ".join(f"{k}={S.eig(k):.3f}" for k in names))
print(f"{'eps':>5} | {'REIG pick':>9} | {'Gauss-worst pick':>16} | {'worst-bound pick':>16} | REIG s/b | bound s/b")
for r in rows:
    pick = lambda a, b: names[int(b > a)]
    print(f"{r[0]:>5} | {pick(r[1], r[5]):>9} | {pick(r[2], r[6]):>16} | {pick(r[4], r[8]):>16} |"
          f" {r[1]:.3f}/{r[5]:.3f} | {r[4]:.3f}/{r[8]:.3f}")
print("REIG >= worst bound - eps check:",
      all(bool(np.all(rows[:, 1 + 4 * j] - rows[:, 4 + 4 * j] <= rows[:, 0] + 1e-6)) for j in range(2)))
print("REIG >= worst bound check:          ",
      all(bool(np.all(rows[:, 1 + 4 * j] >= rows[:, 4 + 4 * j] - 1e-6)) for j in range(2)))
print("\nworst-case regret if the real prior is within KL kappa:")
for r in regret:
    print(f"  kappa={r[0]:<5} sharp={r[1]:.3f}  broad={r[2]:.3f}  -> safer: {names[int(r[2] < r[1])]}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
th = S.TH
for k, c in zip(names, ["C3", "C0"]):
    axes[0].plot(th, S.mean_fn(th, k), color=c, label=f"{k} sensor (w={S.SENSORS[k][1]})")
axes[0].plot(th, S.prior_weights() / S.prior_weights().max(), "k:", label="prior p (scaled)")
q_tilt = S.reig_tilt_prior("sharp", 0.2)[0]
axes[0].plot(th, q_tilt / S.prior_weights().max(), "C3--", lw=1, label="worst prior for sharp, eps=0.2 (scaled)")
axes[0].set_xlim(-4, 4); axes[0].set_xlabel("theta"); axes[0].set_ylabel("mean of y")
axes[0].set_title("Two sensors and the prior that hurts the sharp one", fontsize=10)
axes[0].legend(fontsize=7)
for j, (k, c) in enumerate(zip(names, ["C3", "C0"])):
    axes[1].plot(rows[:, 0], rows[:, 1 + 4 * j], "-", color=c, lw=2, label=f"REIG, {k}")
    axes[1].plot(rows[:, 0], rows[:, 2 + 4 * j], ":", color=c, label=f"worst Gaussian prior, {k}")
    axes[1].plot(rows[:, 0], rows[:, 3 + 4 * j], "--", color=c, label=f"real EIG at REIG's worst prior, {k}")
axes[1].set_xlabel("eps"); axes[1].set_ylabel("nats"); axes[1].legend(fontsize=6)
axes[1].set_title("REIG flips at eps ~ 0.15; Gaussian priors never flip", fontsize=10)
axes[2].plot(regret[:, 0], regret[:, 1], "C3o-", label="always pick sharp")
axes[2].plot(regret[:, 0], regret[:, 2], "C0o-", label="always pick broad")
axes[2].set_xlabel("kappa = KL size of the real prior error"); axes[2].set_ylabel("worst-case regret (nats)")
axes[2].set_title("Worst-case regret: sharp is safer at every kappa", fontsize=10); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RESULTS}/05_competing_designs.png", dpi=150)
