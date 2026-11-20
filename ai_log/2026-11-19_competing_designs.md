# AI session - 2026-11-19 - Ben (with Claude)

**Tool / model:** Claude

**Goal:** Build a design problem where the best design changes with eps (the
linear-Gaussian model can't do this), then test whether eps calibration leads
to better decisions. Suggested independently by Claude and the second AI reviewer.

**What the AI produced:** `src/sensors.py` (y ~ N(tanh((theta-c)/w), 0.05^2),
theta ~ N(0,1)), experiments 05 and 06.

**How we checked it:**
- First sensor menu had one design winning at every eps (no flip). Claude ran a
  small parameter search and kept a setting with a flip: sharp (w=0.5) vs broad (w=2.5).
- All Exp 05 values use a fine grid (no Monte Carlo). Checked REIG >= our best
  bound on the true worst case, and REIG - bound <= eps (paper eq. 15-16). Both hold.
- A crash in a summary check (array truth value) was fixed; a misleading plot
  title ("safer choice depends on kappa") was corrected after looking at the numbers.

**Results (one toy model; preliminary):**
- Plain EIG prefers sharp (2.096 vs 1.943). REIG switches to broad at eps ~ 0.15-0.2.
- Worst case over GAUSSIAN priors (mean/scale wrong) never switches. But the real
  EIG at REIG's own worst prior (an exponential tilt, bimodal for the sharp sensor)
  is much lower than any Gaussian prior gives, and taking the lower of the two the
  preference switches between eps = 0.1 and 0.2, consistent with REIG.
  Lesson: checking only "mean/variance is wrong" misspecification misses the danger.
- Worst-case REGRET (over the priors we checked) favours sharp at every kappa.
  So "maximise the worst-case EIG" (what REIG does) and "minimise worst-case regret"
  give different answers. Which robustness notion is right is a modelling choice.
  Caveat: regret was maximised over a limited set (Gaussian ball + two tilts).
- Exp 06: with tiny inner budgets NMC is wildly biased (M=5: +19 nats for sharp).
  The crossing rule's EIG estimates are nearly unbiased already at M=20
  (errors +0.00 / -0.02), and it picks the right sensor 87% (M=20) and 100% (M=100)
  of runs vs 77% / 93% for plain NMC. Contrastive alone is similar at M>=20.
  Note: "REIG eps=0.2" picking broad is not an error in the robust sense; at that
  radius the robust optimum is broad.

**Next steps:** more seeds / N1 for Exp 06 (30 seeds is noisy); try a richer
family of misspecified priors for the regret question; write up Exp 03 + 05 as
the main story for the slides.
