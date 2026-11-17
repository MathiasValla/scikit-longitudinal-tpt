# Scikit-longitudinal’s TpT splitter: growing decision trees along the
time dimension of longitudinal data
Mathias Valla, Simon Provost

*How to fit and read a Time-penalized Tree in scikit-longitudinal, using
public follow-up data on primary biliary cholangitis.*

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

Two patients can have the same bilirubin level at their last visit, even
though it has been rising in one patient and falling in the other. A tree
fitted to the last visit sees the same bilirubin value for both.
A Time-penalized Tree (TpT) chooses a measurement and a visit at each
split. It favors earlier visits when the gains are close and can send
shorter records to a separate leaf. We show how
to fit and read such a tree in scikit-longitudinal. In public Mayo Clinic
data, we use the first five years of follow-up to predict death or liver
transplantation between years five and ten. The fitted tree asks for
albumin and bilirubin at different visits and has two leaves for patients
who lack the requested visit. Across ten repeats of five-fold
cross-validation in this cohort, mean out-of-fold ROC AUC was 0.815 for
TpT and 0.723 for a CART tree given only the last available visit.

### Repository layout

- `main.qmd`: Quarto source for the article.
- `scripts/reproduce.sh`: creates the article environment and renders HTML and PDF.
- `environment.yml`: frozen Python environment used for the examples.
- `setup-env-ci.sh`: environment-setup hook for Computo's GitHub Actions workflow.
- `setup-render-ci.sh`: render-step hook for Computo's GitHub Actions workflow.
- `scripts/build_tpt_backend.py`: rebuilds the pinned backend with the
  article's TpT duration-routing correction.
- `analysis/tpt_case_study.py`: prepares the landmark cohort, fits and
  evaluates the models, and draws the figures.
- `data/raw/pbcseq.csv`: public data snapshot used in the article.

### Reproducibility

With Conda, Quarto and LaTeX available, run `bash scripts/reproduce.sh`.
The script builds the article's backend and renders HTML and PDF against
the included data snapshot. To use an already downloaded copy of the
backend source archive, set `TPT_BACKEND_SDIST` to its path; the script
checks the SHA-256 digest whether the archive is local or downloaded.
