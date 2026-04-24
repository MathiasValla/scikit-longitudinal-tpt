# Scikit-longitudinal’s TpT splitter: growing decision trees along the
time dimension of longitudinal data
Mathias Valla, Simon Provost
2026-04-23

*A reproducible article presenting the Time-penalized Trees (TpT)
splitter implemented in scikit-longitudinal, with one classification
example on the primary biliary cirrhosis longitudinal dataset and one
regression example on the Produc US state production panel.*

[![build and
publish](https://github.com/MathiasValla/scikit-longitudinal-tpt/actions/workflows/build.yml/badge.svg)](https://github.com/MathiasValla/scikit-longitudinal-tpt/actions/workflows/build.yml)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Title

Scikit-longitudinal’s TpT splitter: growing decision trees along the
time dimension of longitudinal data

### Authors

- Mathias Valla (Chaire DIALog, Institut Louis Bachelier, Paris, France)
- Simon Provost (School of Computing, University of Kent)

### Abstract

We present the Time-penalized Trees (TpT) splitter as implemented in
scikit-longitudinal, a Python package for longitudinal machine learning.
TpT extends recursive partitioning to repeated-measures covariates by
penalizing candidate splits that rely on later observation waves. We
describe the underlying split criterion, summarize the main user-facing
parameters, and illustrate the implementation on two public longitudinal
datasets: a subject-level classification problem based on the primary
biliary cirrhosis follow-up data and a state-level regression problem
based on the Produc panel. These examples show how the time penalty
affects the selected split times and how the exported trees reveal both
which predictors matter and when they matter.

### Repository layout

- `paper.qmd`: Quarto source for the article.
- `environment.yml`: frozen Python environment used for the examples.
- `setup-env-ci.sh`: environment-setup hook for Computo's GitHub Actions workflow.
- `setup-render-ci.sh`: render-step hook for Computo's GitHub Actions workflow.
- `scripts/prepare_example_data.py`: downloads and preprocesses the open
  datasets.
- `analysis/tpt_examples.py`: helper functions used by the paper to run
  the two TpT examples.
- `data/processed/`: processed subject-level longitudinal datasets used
  in the paper.

### Reproducibility 

1.  Create and activate the environment described in `environment.yml`.
2.  Run `python scripts/prepare_example_data.py` from the activated
    environment.
3.  Ensure Quarto is available locally. The repository vendors the
    Computo extension under `_extensions/`.
4.  The root-level `setup-env-ci.sh` and `setup-render-ci.sh` scripts
    are the two customization hooks used by Computo’s GitHub Actions
    workflow.
5.  Render the HTML article with
    `quarto render paper.qmd --to computo-html`, or use
    `bash scripts/render_local.sh computo-html`.
6.  Render the PDF article with
    `quarto render paper.qmd --to computo-pdf`, or use
    `bash scripts/render_local.sh computo-pdf`.
