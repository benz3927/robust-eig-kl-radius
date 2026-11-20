# Calibrating the KL Radius in Robust Expected Information Gain

Data Mining (16:960:588), Rutgers, Fall 2026 - final project, research question Q01.

Based on: Go, J. and Isaac, T. (2022). *Robust Expected Information Gain for Optimal
Bayesian Experimental Design Using Ambiguity Sets.* UAI 2022. arXiv:2205.09914.

## Question
Expected information gain (EIG) ranks experiments, but it depends on the prior.
Go & Isaac define robust EIG (REIG): the minimum, over all priors q with
KL(q || p) <= eps, of an **affine (tangent) approximation** of EIG (their eq. 11, 14).
This is not the same as the true worst-case EIG over the ball (their eq. 9), but
they show EIG >= REIG >= true worst case, and the gap to the true worst case is at most eps.

How should the radius `eps` be chosen? Too small gives little protection against a
wrong prior or a noisy estimator; too large makes the design overly cautious.
The paper leaves this open (Sec. 8).

## Layout
```
src/eig.py            EIG / REIG estimators, closed forms for the linear-Gaussian model
src/binary_tests.py   the paper's two-blood-test example, computed exactly
experiments/          one script per experiment, fixed seeds
results/              CSVs and figures written by the experiments
ai_log/               one note per AI session: prompt, output, how we checked it
docs/                 the blog (GitHub Pages)
```

## Experiments
| Script | Question |
|---|---|
| `00_baseline_linear_gaussian.py` | Does sampled REIG (Algorithm 1) match the exact closed form? |
| `01_inner_noise.py` | How does inner-estimator noise (N2, M) bias sampled REIG at each eps? |
| `02_bounds_check.py` | Do the paper's bounds hold, and how far is REIG from a worst-case prior? |
| `03_two_tests.py` | The paper's two-test example: REIG vs true worst case as eps grows |
| `04_eps_calibration.py` | Exploratory rule: choose eps where REIG of an upper-bound estimator meets REIG-max of a lower-bound estimator |

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
for f in experiments/0*.py; do python "$f"; done
```

## Notes on the estimators
- `d_samples_nmc` shares one set of M prior samples across all outcomes when
  estimating p(y). This is valid but makes errors in different d_i correlated.
- "Exact" REIG in the experiments uses a closed form for the Gaussian model,
  so it has no Monte Carlo error at all.

## Team workflow
- Work on a branch, merge through a pull request.
- Fix random seeds; note the exact command that made each figure.
- Log every AI session in `ai_log/` (copy `ai_log/TEMPLATE.md`).
