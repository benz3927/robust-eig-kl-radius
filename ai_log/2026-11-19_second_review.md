# AI session - 2026-11-19 (evening) - Ben (second AI reviewer, Claude fixes)

**Tools / models:** a second AI model re-ran Exp 04-06 in a temporary folder and
reviewed the claims; Claude made the fixes.

**Verified by the reviewer:** Exp 04, 05 and 06 reproduce the saved CSVs exactly.
Crossing-rule RMSE 0.0606; REIG switches between eps 0.15 and 0.2; the sharp
sensor's tilted prior is two-humped; Exp 06 numbers match.

**Corrections requested, all accepted:**
1. Exp 06 measures picking the NOMINAL-best sensor (estimation accuracy under p),
   not protection against a wrong prior. Wording fixed in script, plot, blog, outline.
2. The regret result covers only the priors we tested (Gaussian ball + two tilts).
   Plot title and blog no longer call it a global worst case.
3. The Exp 05 "bound check" label was wrong: it checks REIG - (min over tested
   priors) <= eps, which alone does not establish the true worst case. Relabelled.
4. Exp 04 and Exp 06 had separate crossing-rule copies with different fallbacks when
   the curves are already crossed at eps = 0. Now one shared function,
   `src/eig.py: crossing_estimate` (fallback: average of the two means).
   Exp 04 results unchanged after the switch.

**New result suggested by the reviewer:** bracket the true worst case as
[REIG - eps, min over tested priors] (paper eq. 16 + feasibility). Computed for all eps:
sharp better at eps 0.02-0.1, overlap at 0.15, broad better at 0.2-0.4. So REIG's
switch is confirmed for the true worst case, within grid accuracy.
Saved in `results/05_true_worst_intervals.csv`.

**Third pass (blog review, 10:58 PM):** the reviewer checked the live blog (figures
load, links resolve, interval table matches) and asked for wording fixes, all accepted:
- "every quantity can be checked exactly" -> "closed-form checks or numerical reference
  calculations" (the sensor model uses quadrature)
- NMC over- / contrastive under-estimation qualified as "in expectation"
- "both move toward the truth" -> "the two estimates move toward each other; the
  crossing point need not equal the true EIG"
- AI-process description corrected: the second model DID see the first model's pasted
  reasoning, and reproduction means "first run in the AI's cloud workspace, then re-run
  on Ben's laptop with identical numbers"
Still open: team names and each member's contribution line.
