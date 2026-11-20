# AI session - 2026-11-19 - Ben (with Claude)

**Tool / model:** Claude

**Goal:** Write a first REIG implementation and check it against Go & Isaac (2022).

**What the AI produced (v1):** `kl_worst_case_mean`, which applied the KL-ball dual
to the per-(theta, y) terms log p(y|theta)/p(y).

**How we checked it:** Ran it on the linear-Gaussian model. Robust values went
negative for eps = 0.5 at every design, which cannot happen for an information gain.
Read the paper: the KL ball is on the prior p(theta) only (Sec. 2.1, 4), and the dual
(eq. 18, Algorithm 1) is applied to d(theta) = KL(p(y|theta) || p(y)), which is >= 0.
v1 was actually the joint-ambiguity criterion of eq. (20), which the paper says is
more conservative and does not study.

**Verdict:** changed. v2 follows Algorithm 1: N1 prior samples, N2 outcomes each,
average to get d_i, then solve the 1-D dual.

**What we learned / next step:**
- With exact d, REIG stays >= 0.
- With sampled d_i (N2 = 10), REIG can still dip slightly below 0 at large eps.
  The true d_i are >= 0, so any negative value must come from estimation error,
  and a large eps puts more weight on the lowest d_i estimates (paper Sec. 6.3).
  (An earlier draft of this note said "the smallest estimate is the noisiest";
  that was not established and was removed after review.)
