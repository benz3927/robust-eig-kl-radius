# AI session - 2026-11-19 - Ben (with Claude)

**Tool / model:** Claude

**Goal:** Write a first REIG implementation and check it against Go & Isaac (2022).

**What the AI produced (v1):** `kl_worst_case_mean`, which applied the KL-ball dual
to the per-(theta, y) terms log p(y|theta)/p(y).

**How we checked it:** Ran it on the linear-Gaussian model. Robust values went
negative for eps = 0.5 at every design, which cannot happen for an information gain.
Read the paper: the KL ball is on the prior p(theta) only (Sec. 2.1, 4), and the dual
(eq. 18, Algorithm 1) is applied to d(theta) = KL(p(y|theta) || p(y)), which is >= 0.
v1 was actually the joint-ambiguity criterion of eq. (20), which the paper notes is
more conservative and does not study.

**Verdict:** changed. v2 follows Algorithm 1: N1 prior samples, N2 outcomes each,
average to get d_i, then solve the 1-D dual. Added exact d(theta) for the Gaussian
model as a check.

**What we learned / next step:**
- With exact d, REIG stays >= 0 and shrinks smoothly as eps grows.
- With sampled d_i (N2 = 10), REIG can still dip slightly below 0 at large eps.
  Large eps pushes the value toward min_i d_i, and the smallest estimate is also the
  noisiest. So eps interacts with inner-estimator noise. This links directly to the
  paper's open question (Sec. 8): choose eps on the order of the estimator error.
