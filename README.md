# Calibrating the KL Radius in Robust Expected Information Gain

Data Mining (16:960:588), Rutgers, Fall 2026 - final project, research question Q01.

Based on: Go, J. and Isaac, T. (2022). *Robust Expected Information Gain for Optimal
Bayesian Experimental Design Using Ambiguity Sets.* UAI 2022.

## Question
Robust EIG replaces the expected information gain with its worst case over all
distributions within a KL ball of radius `eps` around the nominal model.
How should `eps` be chosen? Too small gives little protection against a wrong model;
too large makes the design overly cautious.

## Layout
```
src/           shared code (EIG estimators, KL-ball worst case)
experiments/   one script per experiment, fixed seeds
results/       small CSVs and figures
ai_log/        one note per AI session: prompt, output, how we checked it
docs/          the blog (GitHub Pages)
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/00_baseline_linear_gaussian.py
```

## Team workflow
- Work on a branch, merge through a pull request.
- Fix random seeds; note the exact command that made each figure.
- Log every AI session in `ai_log/` (copy `ai_log/TEMPLATE.md`).
