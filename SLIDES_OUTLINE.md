# Slide outline - Dec 7 presentation (15-20 min)

Working title: **How Robust Is Robust Enough? Calibrating the KL Radius in Robust EIG**

Rough timing: about 1-1.5 min per slide, 13 slides. Figures are in `results/`.

---

## Part 1 - Problem (about 4 min)

**1. Title**
- Title, team names, Q01, repo link

**2. Choosing experiments with EIG**
- What EIG measures, in one sentence and one formula
- Why it matters: picks the experiment that teaches us the most
- Speaker note: keep the math light; one picture of prior -> outcome -> posterior

**3. The catch: EIG depends on the prior**
- The paper's two-blood-test example: same EIG at a 50% prior, different EIG nearby
- Figure: `03_two_tests.png`, left panel
- Sampling error behaves like a perturbed prior too

**4. Robust EIG (Go & Isaac 2022)**
- Worst case over priors within KL radius eps
- Uses a tangent (affine) approximation, which becomes a cheap 1-D dual problem
- The open question, in their words: no a priori eps "seems unlikely in most cases"
- **Our question: how should eps be chosen?**

## Part 2 - Approach and AI workflow (about 4 min)

**5. Our approach**
- Small models where every quantity can be checked exactly:
  linear-Gaussian, two tests, sharp vs broad sensor
- Over- vs under-estimating EIG estimators (NMC vs contrastive)

**6. How we used AI**
- Diagram: generate -> check against theory/paper -> second-AI review -> reproduce -> log
- Tools used, and who checked what [team fills in]

**7. What the checks caught** (the slide Hu will care most about)
- Wrong definition: KL ball on (theta, y) instead of theta, gave negative "information"
- Overclaiming caught by a second AI
- A sign bug caught by a jump in a baseline number (0.04 -> 1.18)
- A misleading plot title the numbers did not support
- A second review narrowed four claims to what the evidence supports
- Figure: `00_baseline.png` (right panel shows the wrong version)

## Part 3 - Results (about 6 min)

**8. eps and estimator noise are tangled**
- Noise in each d_i pulls REIG down; N2 matters more than M
- Figure: `01_inner_noise.png`
- Takeaway: eps can't be chosen without knowing the estimator's noise

**9. REIG can miss sensitivity**
- Two tests: REIG says test B loses nothing (flat line); its real worst case drops by half
- Figure: `03_two_tests.png`, right panel
- Why: the tangent approximation drops the curvature term

**10. ...and it can catch a danger simple checks miss**
- Sharp vs broad sensor; REIG flips at eps ~ 0.15-0.2
- Gaussian-only checks never flip; REIG's own worst prior is two-humped
- Bracketing the TRUE worst case ([REIG - eps, min over tested priors]) confirms the flip:
  sharp better up to eps = 0.1, broad from eps = 0.2 (0.15 undecided)
- Twist: by regret over the priors we tested, sharp is always lower
- Figure: `05_competing_designs.png`

**11. A data-driven radius**
- Crossing rule: REIG of NMC meets REIG-max of contrastive
- RMSE 0.061 vs 0.099-0.246 for the alternatives (linear-Gaussian, M = 10)
- Sensor model (100 seeds), picking the NOMINAL-best sensor (estimation accuracy, not robustness): crossing rule nearly unbiased from M = 20, best pick rate at M = 20 (88% vs 75% NMC)
- Honest caveat: at M = 5 it picks worse than NMC (58% vs 78%); NMC is right for the wrong reason
- Figures: `04_eps_calibration.png`, `06_noisy_design_choice.png`

**11b. Main result: a calibrated radius for sampling noise**
- Penalty is a variance penalty: REIG ~ EIG - sqrt(2 eps Var d)  (known DRO expansion, Duchi-Glynn-Namkoong 2021)
- Match to a confidence bound: eps = z^2 / (2N)  (e.g. 0.0027 for N = 500)
- Interval [REIG(PCE), REIG_max(NMC)] covers EIG 92-100% at any M; naive NMC interval 0-89%
- Combined rule for prior + sampling: eps_total = (sqrt(eps_p) + sqrt(eps_N))^2, coverage 96-98%
- Paired test picks the right sensor up to 100% of runs, wrong sensor in 0 of 2,700 runs
- Figures: `07_calibrated_radius.png`, `08_paired_certified_choice.png`
- What stays open: how big eps_p should be (modelling judgment); finite-sample correction for skewed d

**11c. Theory: the full answer, under a stated model**
- Thm 1: REIG = EIG - sqrt(2 eps Var d) + kappa3 eps/(3 sigma^2) + ...  (proof + sympy check, Exp 10)
- Thm 2: eps_N = z^2/(2N) -> calibrated lower bound.  Thm 3: PCE/NMC interval valid for any M
- Thm 4: prior fitted from n0 points, k params -> eps_p = chi2_k/(2 n0); bound holds under the TRUE prior,
  for ALL designs at once (held in 99.7-100% of runs; plug-in EIG 3.5-66%)
- Prop 6: no rule for eps_p can have a guarantee without information about the prior's error
  -> explains Go & Isaac's "a definite a priori estimate seems unlikely"
- Figure: `09_theory_checks.png`.  Proofs are in the blog (collapsible)

## Part 4 - Wrap-up (about 2-3 min)

**12. What we learned**
- eps does two jobs: prior error and estimator noise - and the noise job now has a principled answer
- The tangent approximation cuts both ways
- "Robust" needs a definition (max-min vs regret)
- About AI: fast at generating, but every real error was caught by an independent check, not by the AI itself

**13. Limitations and next steps**
- Toy, 1-D models only; not yet the paper's benchmarks
- No theory yet for the crossing rule
- Next: paper benchmarks, scaling of eps* with M and N2

---

## To decide at the team meeting
- Who presents which part
- Whether to show formulas on slide 4 or keep it visual
- Likely questions: why KL and not another distance? Why trust a toy model? What would you do with one more month?
