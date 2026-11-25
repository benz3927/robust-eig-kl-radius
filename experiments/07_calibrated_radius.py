"""Exp 07 - A calibrated KL radius: does eps = z^2/(2N) make REIG a valid confidence bound?

Claims tested (each could fail):
  A. Expansion: REIG_eps = EIG - sqrt(2 eps Var(d)) + O(eps)   (linear-Gaussian, exact)
  B. With exact d(theta) (outer sampling noise only), REIG_hat(eps = z^2/(2N)) is a
     95% lower confidence bound for EIG. Compared with eps = 0 and a fixed eps = 0.05.
  C. With inner noise (finite M), the interval [REIG(PCE, eps_N), REIG_max(NMC, eps_N)]
     covers EIG with >= 90% probability for ANY M. Compared with a naive NMC CLT interval.
  D. Decisions: call "design A beats B" only when A's lower bound > B's upper bound.
     How often is such a call wrong (should be <= ~5%) and how often is a call made?
  E. Combining prior uncertainty eps_p with sampling noise:
     eps_total = (sqrt(eps_p) + sqrt(eps_stat))^2 as a 95% lower bound for REIG_{eps_p}.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
import src.sensors as S
from src.eig import (eig_closed_form, reig_closed_form, d_closed_form, d_samples_indep,
                     reig_dual, eps_stat, eps_combined, eig_interval, _d_coeffs)
from scipy.stats import norm

Z = norm.ppf(0.95)
rng0 = np.random.default_rng(2026)
out = {}

# ---------------- A. expansion ----------------
eps_grid = np.logspace(-4, -0.5, 30)
A = []
for x in [1.0, 2.0]:
    a, b = _d_coeffs(x)
    for e in eps_grid:
        A.append([x, e, reig_closed_form(x, e), a + b - 2 * b * np.sqrt(e)])
A = np.array(A)
np.savetxt(f"{RESULTS}/07a_expansion.csv", A, delimiter=",", comments="", header="x,eps,reig_exact,first_order")

# ---------------- models with exact d ----------------
def lg_exact(x):
    return lambda n, rng: d_closed_form(rng.normal(size=n), x), eig_closed_form(x)

def sensor_exact(name):
    return lambda n, rng: S.d_exact_at(rng.normal(size=n), name), S.eig(name)

models = {"linear-Gaussian x=1": lg_exact(1.0), "linear-Gaussian x=3": lg_exact(3.0),
          "sensor sharp": sensor_exact("sharp"), "sensor broad": sensor_exact("broad")}

# ---------------- B. coverage, outer noise only ----------------
Ns, REPS_B = [25, 100, 400, 1600], 1000
B = []
for mi, (mname, (draw, truth)) in enumerate(models.items()):
    for N in Ns:
        hits = {"eps=0": 0, "calibrated": 0, "eps=0.05": 0}
        gap = {k: [] for k in hits}
        for r in range(REPS_B):
            d = draw(N, np.random.default_rng(10_000 * mi + 100 * N + r))
            vals = {"eps=0": d.mean(), "calibrated": reig_dual(d, eps_stat(N))[0],
                    "eps=0.05": reig_dual(d, 0.05)[0]}
            for k, v in vals.items():
                hits[k] += v <= truth
                gap[k].append(truth - v)
        for k in hits:
            B.append([mi, N, ["eps=0", "calibrated", "eps=0.05"].index(k), hits[k] / REPS_B, np.mean(gap[k])])
B = np.array(B)
np.savetxt(f"{RESULTS}/07b_coverage_exact_d.csv", B, delimiter=",", comments="",
           header="model_index,N,method(0=eps0,1=calibrated,2=eps0.05),coverage,mean_truth_minus_bound")

# ---------------- C. interval with inner noise ----------------
def lg_noisy(x):
    def f(n, m, contrastive, seed):
        return d_samples_indep(x, n1=n, m=m, contrastive=contrastive, rng=seed)[1]
    return f, eig_closed_form(x)

def sensor_noisy(name):
    def f(n, m, contrastive, seed):
        return S.d_samples_indep(name, n1=n, m=m, contrastive=contrastive, rng=seed)
    return f, S.eig(name)

noisy = {"linear-Gaussian x=2": lg_noisy(2.0), "sensor sharp": sensor_noisy("sharp"),
         "sensor broad": sensor_noisy("broad")}
Ms, NsC, REPS_C = [5, 20, 100], [100, 400], 300
C = []
store = {}
for mi, (mname, (draw, truth)) in enumerate(noisy.items()):
    for m in Ms:
        for N in NsC:
            cov_new = cov_naive = 0
            widths_new, widths_naive = [], []
            for r in range(REPS_C):
                seed = 1_000_000 * mi + 10_000 * m + 10 * N + r
                dl = draw(N, m, True, seed)
                du = draw(N, m, False, seed + 5_000_000)
                lo, hi = eig_interval(dl, du)
                store[(mname, m, N, r)] = (lo, hi)
                se = du.std(ddof=1) / np.sqrt(N)
                nlo, nhi = du.mean() - Z * se, du.mean() + Z * se
                cov_new += lo <= truth <= hi
                cov_naive += nlo <= truth <= nhi
                widths_new.append(hi - lo); widths_naive.append(nhi - nlo)
            C.append([mi, m, N, cov_new / REPS_C, np.median(widths_new), cov_naive / REPS_C, np.median(widths_naive)])
C = np.array(C)
np.savetxt(f"{RESULTS}/07c_interval_inner_noise.csv", C, delimiter=",", comments="",
           header="model_index,M,N,coverage_new,median_width_new,coverage_naive_nmc,median_width_naive")

# ---------------- D. certified design decisions (sharp vs broad) ----------------
D = []
for m in Ms:
    for N in NsC:
        wrong = right = 0
        naive_wrong = 0
        for r in range(REPS_C):
            ls, hs = store[("sensor sharp", m, N, r)]
            lb, hb = store[("sensor broad", m, N, r)]
            if ls > hb: right += 1            # truth: sharp is better (2.096 vs 1.943)
            elif lb > hs: wrong += 1
        D.append([m, N, right / REPS_C, wrong / REPS_C])
D = np.array(D)
np.savetxt(f"{RESULTS}/07d_certified_decisions.csv", D, delimiter=",", comments="",
           header="M,N,frac_correct_certified,frac_wrong_certified")

# ---------------- E. combining prior radius and sampling radius ----------------
E = []
REPS_E = 1000
for x in [1.0, 2.0]:
    for ep in [0.01, 0.05, 0.2]:
        target = reig_closed_form(x, ep)
        for N in Ns:
            hit_comb = hit_plug = 0
            for r in range(REPS_E):
                d = d_closed_form(np.random.default_rng(7_000_000 + 1000 * N + r + int(1e5 * ep) + int(1e8 * x)).normal(size=N), x)
                hit_comb += reig_dual(d, eps_combined(ep, N))[0] <= target
                hit_plug += reig_dual(d, ep)[0] <= target
            E.append([x, ep, N, hit_comb / REPS_E, hit_plug / REPS_E])
E = np.array(E)
np.savetxt(f"{RESULTS}/07e_combined_radius.csv", E, delimiter=",", comments="",
           header="x,eps_prior,N,coverage_combined,coverage_plugin_eps_prior")

# ---------------- print ----------------
names = list(models)
print("A. max relative error of first-order expansion for eps <= 0.01:",
      f"{np.max(np.abs(A[A[:,1]<=0.01,2]-A[A[:,1]<=0.01,3])/A[A[:,1]<=0.01,2]):.4f}")
print("\nB. one-sided coverage of EIG (target 0.95), exact d:")
for mi, mn in enumerate(names):
    for N in Ns:
        s = B[(B[:, 0] == mi) & (B[:, 1] == N)]
        print(f"  {mn:<22} N={N:<5} eps=0: {s[0,3]:.3f}  calibrated: {s[1,3]:.3f} (gap {s[1,4]:.3f})  eps=0.05: {s[2,3]:.3f} (gap {s[2,4]:.3f})")
print("\nC. two-sided coverage (target >= 0.90) with inner noise:")
for row in C:
    print(f"  {list(noisy)[int(row[0])]:<22} M={int(row[1]):<4} N={int(row[2]):<4} new: {row[3]:.3f} (width {row[4]:.3f})  naive NMC: {row[5]:.3f} (width {row[6]:.3f})")
print("\nD. certified sharp-vs-broad calls:")
for row in D:
    print(f"  M={int(row[0]):<4} N={int(row[1]):<4} correct certified: {row[2]:.3f}  wrong certified: {row[3]:.3f}")
print("\nE. coverage of REIG_{eps_p} (target 0.95):")
for row in E:
    print(f"  x={row[0]} eps_p={row[1]:<5} N={int(row[2]):<5} combined: {row[3]:.3f}  plug-in eps_p only: {row[4]:.3f}")

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
for x, c in [(1.0, "C0"), (2.0, "C3")]:
    s = A[A[:, 0] == x]
    ax[0, 0].loglog(s[:, 1], s[:, 2][0] * 0 + eig_closed_form(x) - s[:, 2], "-", color=c, label=f"EIG - REIG, x={x}")
    ax[0, 0].loglog(s[:, 1], eig_closed_form(x) - s[:, 3], "--", color=c, label=f"sqrt(2 eps Var d), x={x}")
ax[0, 0].set_xlabel("eps"); ax[0, 0].set_ylabel("robustness penalty (nats)")
ax[0, 0].set_title("A. Penalty grows like sqrt(eps) for small eps", fontsize=10); ax[0, 0].legend(fontsize=7)
for mi, mn in enumerate(names):
    s = B[(B[:, 0] == mi) & (B[:, 2] == 1)]
    ax[0, 1].plot(s[:, 1], s[:, 3], "o-", label=mn)
ax[0, 1].axhline(0.95, color="k", ls=":", label="target 0.95")
ax[0, 1].set_xscale("log"); ax[0, 1].set_ylim(0.8, 1.0)
ax[0, 1].set_xlabel("N (prior samples)"); ax[0, 1].set_ylabel("P(lower bound <= EIG)")
ax[0, 1].set_title("B. Calibrated eps = z^2/(2N): coverage with exact d", fontsize=10); ax[0, 1].legend(fontsize=7)
xt = np.arange(len(C))
ax[1, 0].bar(xt - 0.2, C[:, 3], 0.4, label="new interval [REIG(PCE), REIG_max(NMC)]")
ax[1, 0].bar(xt + 0.2, C[:, 5], 0.4, label="naive NMC +/- z se")
ax[1, 0].axhline(0.90, color="k", ls=":")
ax[1, 0].set_xticks(xt, [f"{['LG','sharp','broad'][int(r[0])]}\nM={int(r[1])}\nN={int(r[2])}" for r in C], fontsize=6)
ax[1, 0].set_ylabel("two-sided coverage"); ax[1, 0].set_ylim(0, 1.05)
ax[1, 0].set_title("C. With inner noise: target >= 0.90", fontsize=10); ax[1, 0].legend(fontsize=7)
for x, mk in [(1.0, "o"), (2.0, "s")]:
    for ep, c in [(0.01, "C0"), (0.05, "C1"), (0.2, "C2")]:
        s = E[(E[:, 0] == x) & (E[:, 1] == ep)]
        ax[1, 1].plot(s[:, 2], s[:, 3], mk + "-", color=c, label=f"x={x}, eps_p={ep}")
ax[1, 1].axhline(0.95, color="k", ls=":")
ax[1, 1].set_xscale("log"); ax[1, 1].set_ylim(0.8, 1.0)
ax[1, 1].set_xlabel("N"); ax[1, 1].set_ylabel("P(bound <= REIG_eps_p)")
ax[1, 1].set_title("E. Combined radius (sqrt(eps_p)+sqrt(eps_stat))^2", fontsize=10); ax[1, 1].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(f"{RESULTS}/07_calibrated_radius.png", dpi=150)
