# AI session - 2026-11-19 - Ben (Claude + second AI reviewer)

**Tools / models:** Claude wrote code; a second AI reviewed it independently.

**Goal:** Verify the v2 REIG code, then run first experiments on choosing eps.

**Review findings (second AI), all accepted:**
1. README blurred REIG (affine approximation, eq. 14) with true worst-case EIG (eq. 9). Fixed.
2. "Exact d_i" still used 500 sampled theta values. Replaced by a full closed form
   for the Gaussian model (no Monte Carlo at all).
3. The claim "the smallest estimate is the noisiest" was overstated. Reworded.
4. Exact-d_i results were saved but not plotted. Now plotted in Exp 00.
5. Shared inner samples across outcomes were undocumented. Now documented.
The reviewer also checked the dual solver against a constrained optimisation (agreed to 5e-12).

**Bug caught during this session:** while refactoring, the eps = 0 shortcut in
`reig_dual` returned -mean(d). Caught because Exp 00's error jumped from 0.04 to
1.18 and Exp 04 picked eps* = 0 every time. Fixed; numbers back to the earlier run.

**Checks run:** closed-form REIG vs sampled dual with 2M exact samples (agree to 4 dp);
two-test example reproduces the paper (r = 18.4%, EIG = 0.216 nats).

**Results (first pass, one toy model, treat as preliminary):**
- Exp 01: bias of sampled REIG is driven mainly by N2 (outcomes per theta), much
  less by M. At eps = 0.5, x = 0.5: bias -0.45 (N2=1), -0.12 (N2=5), -0.03 (N2=25).
- Exp 02: EIG >= REIG and REIG - (Gaussian worst case) <= eps hold everywhere tested.
  The gap grows with eps and with design strength (x = 3: 0.08 at eps 0.1, 0.53 at eps 1).
- Exp 03: both REIG and the true worst case prefer test B for every eps > 0.
  But REIG for test B is flat (its d values are equal for both theta, so the affine
  approximation sees no sensitivity), while B's true worst case falls from 0.216 to 0.105
  at eps = 0.3. REIG can hide sensitivity that lives in the curvature term.
- Exp 04: with M = 10, the crossing rule gave RMSE 0.061 vs 0.246 (NMC), 0.115
  (contrastive), 0.099 (average of the two). eps* ranged 0 - 0.074 and grew with x.

**Next steps:** test the crossing rule on a model where the design ranking matters,
vary M, and check whether the eps* it picks also gives good robustness.
