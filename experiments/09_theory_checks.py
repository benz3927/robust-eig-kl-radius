"""Exp 09 - Checks for the theory section (blog Section 6).

A. Theorem 4 (prior fitted from data). True prior q = N(0,1). The designer only sees
   n0 draws from q, fits a Gaussian prior p_hat by maximum likelihood (k = 2 parameters)
   and uses eps_p = chi2_{2,0.95}/(2 n0). Claims:
     (i)  P( KL(q || p_hat) <= eps_p ) -> 0.95
     (ii) P( EIG(q, xi) >= REIG(p_hat, xi, eps_p) - eps_p  for ALL designs ) >= ~0.95
   Compared with the plug-in EIG(p_hat) and with REIG(p_hat, eps_p) without the -eps_p.
   Designs: linear-Gaussian x in {0.5, 1, 2, 3}; sensors sharp and broad.
B. Proposition 5 (sampling error at a fixed prior radius). Delta-method lower bound for
   REIG_{eps_p} vs the first-order combined radius of Exp 07E, with exact d.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import RESULTS
import src.sensors as S
from src.eig import (eig_closed_form, reig_closed_form, d_closed_form, reig_dual,
                     eps_prior_from_fit, reig_lcb_delta, eps_combined)

def kl_gauss(m1, s1, m2, s2):  # KL(N(m1,s1^2) || N(m2,s2^2))
    return np.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5

# ---------------- A ----------------
n0s = [10, 30, 100, 300]
xs = [0.5, 1.0, 2.0, 3.0]
REPS_LG, REPS_S = 2000, 200
A = []
for n0 in n0s:
    ep = eps_prior_from_fit(2, n0)
    inball = cov_lg = cov_lg_noeps = cov_lg_plug = 0
    cov_s = cov_s_noeps = cov_s_plug = 0
    for r in range(REPS_LG):
        obs = np.random.default_rng(90_000 + 10_000 * n0 + r).normal(size=n0)
        m, s = obs.mean(), obs.std()                                    # MLE
        inball += kl_gauss(0, 1, m, s) <= ep
        truth = np.array([eig_closed_form(x, 1.0) for x in xs])
        lb = np.array([reig_closed_form(x, ep, s_theta=s) for x in xs])
        plug = np.array([eig_closed_form(x, s) for x in xs])
        cov_lg += np.all(truth >= lb - ep)
        cov_lg_noeps += np.all(truth >= lb)
        cov_lg_plug += np.all(truth >= plug)
        if r < REPS_S:
            ts = np.array([S.eig(k) for k in ("sharp", "broad")])
            ls = np.array([S.reig(k, ep, m, s) for k in ("sharp", "broad")])
            ps = np.array([S.eig(k, m, s) for k in ("sharp", "broad")])
            cov_s += np.all(ts >= ls - ep); cov_s_noeps += np.all(ts >= ls); cov_s_plug += np.all(ts >= ps)
    A.append([n0, ep, inball / REPS_LG, cov_lg / REPS_LG, cov_lg_noeps / REPS_LG, cov_lg_plug / REPS_LG,
              cov_s / REPS_S, cov_s_noeps / REPS_S, cov_s_plug / REPS_S])
A = np.array(A)
np.savetxt(f"{RESULTS}/09a_prior_fit_radius.csv", A, delimiter=",", comments="",
           header="n0,eps_p,P(true prior in ball),LG_cov_REIG_minus_eps,LG_cov_REIG,LG_cov_plugin,"
                  "sensor_cov_REIG_minus_eps,sensor_cov_REIG,sensor_cov_plugin")
print("A. prior fitted from n0 draws (k=2), eps_p = chi2_2/(2 n0); simultaneous over all designs; target 0.95")
for r in A:
    print(f"  n0={int(r[0]):<4} eps_p={r[1]:.4f} | true prior in ball: {r[2]:.3f} | linear-Gaussian (4 designs): "
          f"REIG-eps {r[3]:.3f}, REIG {r[4]:.3f}, plug-in EIG {r[5]:.3f} | sensors (2): REIG-eps {r[6]:.3f}, "
          f"REIG {r[7]:.3f}, plug-in {r[8]:.3f}")

# ---------------- B ----------------
cases = {"linear-Gaussian x=2": (lambda n, g: d_closed_form(g.normal(size=n), 2.0), lambda e: reig_closed_form(2.0, e)),
         "sensor sharp": (lambda n, g: S.d_exact_at(g.normal(size=n), "sharp"), lambda e: S.reig("sharp", e))}
B = []
REPS_B = 500
for ci, (cname, (draw, target_fn)) in enumerate(cases.items()):
    for ep in [0.05, 0.2, 0.5]:
        target = target_fn(ep)
        for N in [100, 400, 1600]:
            hd = hc = 0; gd = []; gc = []
            for r in range(REPS_B):
                d = draw(N, np.random.default_rng(500_000 * ci + 1000 * N + r + int(1e6 * ep)))
                ld = reig_lcb_delta(d, ep); lc = reig_dual(d, eps_combined(ep, N))[0]
                hd += ld <= target; hc += lc <= target; gd.append(target - ld); gc.append(target - lc)
            B.append([ci, ep, N, hd / REPS_B, np.mean(gd), hc / REPS_B, np.mean(gc)])
B = np.array(B)
np.savetxt(f"{RESULTS}/09b_fixed_radius_sampling.csv", B, delimiter=",", comments="",
           header="case_index,eps_p,N,coverage_delta,mean_gap_delta,coverage_combined,mean_gap_combined")
print("\nB. lower bound for REIG_{eps_p} from N samples; target 0.95")
for r in B:
    print(f"  {list(cases)[int(r[0])]:<20} eps_p={r[1]:<4} N={int(r[2]):<5} delta-method: {r[3]:.3f} (gap {r[4]:.3f})"
          f"  combined radius: {r[5]:.3f} (gap {r[6]:.3f})")

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(A[:, 0], A[:, 2], "ko-", label="true prior inside the ball")
ax[0].plot(A[:, 0], A[:, 3], "C0o-", label="REIG - eps (linear-Gaussian, 4 designs)")
ax[0].plot(A[:, 0], A[:, 6], "C3s-", label="REIG - eps (sensors, 2 designs)")
ax[0].plot(A[:, 0], A[:, 4], "C0o:", label="REIG without - eps (linear-Gaussian)")
ax[0].plot(A[:, 0], A[:, 5], "C0x--", label="plug-in EIG of fitted prior (linear-Gaussian)")
ax[0].axhline(0.95, color="k", ls=":", lw=1)
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.03)
ax[0].set_xlabel("n0 = data points used to fit the prior")
ax[0].set_ylabel("P(bound holds for the TRUE prior, all designs)")
ax[0].set_title("Theorem 4: eps_p = chi2_k / (2 n0)", fontsize=10); ax[0].legend(fontsize=7)
for ci, mk in [(0, "o"), (1, "s")]:
    for ep, c in [(0.05, "C0"), (0.2, "C1"), (0.5, "C2")]:
        s = B[(B[:, 0] == ci) & (B[:, 1] == ep)]
        ax[1].plot(s[:, 2], s[:, 3], mk + "-", color=c, label=f"delta, {list(cases)[ci]}, eps_p={ep}")
        ax[1].plot(s[:, 2], s[:, 5], mk + ":", color=c, alpha=.6)
ax[1].axhline(0.95, color="k", ls=":", lw=1)
ax[1].set_xscale("log"); ax[1].set_ylim(0.75, 1.0)
ax[1].set_xlabel("N"); ax[1].set_ylabel("P(bound <= REIG_eps_p)")
ax[1].set_title("Prop. 5: delta method (solid) vs combined radius (dotted)", fontsize=10)
ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{RESULTS}/09_theory_checks.png", dpi=150)
