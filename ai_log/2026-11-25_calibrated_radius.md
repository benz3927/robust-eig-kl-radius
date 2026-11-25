# AI session - 2026-11-25 - Ben (with Claude)

**Tool / model:** Claude

**Goal:** Ben asked whether the open problem was solved (no), then asked to actually
attempt it. Target: a principled rule for at least one of eps's two jobs.

**Idea (Claude):** for small eps, the KL worst case of a mean is
mean - sqrt(2 eps Var) + O(eps). Matching this to a one-sided confidence bound
mean - z sd/sqrt(N) gives eps = z^2/(2N). Checked against the literature before
building on it: Duchi, Glynn & Namkoong (2021, Math. of OR) prove calibrated
coverage for f-divergence balls of radius rho/n with f''(1)=2; converting to plain
KL (f''(1)=1) gives the same z^2/(2N). So the rule is a known DRO result applied to
REIG, not a new theorem; the new parts are the application to Go & Isaac's question,
the PCE/NMC interval, the combined radius, and the paired certified test.

**How we checked it (Exp 07, 08):**
- A: expansion vs closed form: < 1% relative error for eps <= 0.01.
- B: coverage of calibrated lower bound with exact d, 4 designs x 4 N x 1000 runs:
  94.0-96.7% (target 95). Fixed eps = 0.05 is far too conservative at large N.
- C: interval [REIG(PCE), REIG_max(NMC)] with independent inner samples per theta
  (earlier code shared them, which breaks i.i.d.; changed for this test):
  92-100% coverage in all 18 settings (target >= 90). Naive NMC CLT interval: 0-89%.
- D: separate intervals never separated the two sensors (too wide) -> a negative result.
- Exp 08: paired difference bound with common random numbers: 0 wrong calls in 2,700
  runs; right call in 100% of runs at M=500, N=1600; no call at M <= 20.
- E: combined radius (sqrt(eps_p)+sqrt(eps_N))^2: 95.9-97.8% coverage of REIG_{eps_p}
  (linear-Gaussian only); plugging in eps_p alone: 46-57%.

**Caveats kept in the write-up:** asymptotic guarantee (94.0% for the most skewed
model at N=400); combined rule tested only on the linear-Gaussian model; eps_p for
genuine prior doubt is still a modelling choice. So: "a principled answer for the
sampling-noise half," not "solved."

**Next steps:** bootstrap / finite-sample correction for skewed d; test on the
paper's benchmarks; ideas for eliciting eps_p.
