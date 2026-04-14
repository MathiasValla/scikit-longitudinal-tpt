# Scikit-longitudinal’s TpT splitter: growing decision trees along the
time dimension of longitudinal data
Mathias Valla, Simon Provost
2026-04-13

*A short reproducible software note presenting the Time-penalized Trees
(TpT) splitter in scikit-longitudinal, with one classification example
on the primary biliary cirrhosis longitudinal dataset and one regression
example on the dietox pig growth dataset.*

[![build and
publish](https://github.com/MathiasValla/scikit-longitudinal-tpt/actions/workflows/build.yml/badge.svg)](https://github.com/MathiasValla/scikit-longitudinal-tpt/actions/workflows/build.yml)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Title

Scikit-longitudinal’s TpT splitter: growing decision trees along the
time dimension of longitudinal data

### Authors

- Mathias Valla (Chaire DIALog, Institut Louis Bachelier, Paris, France)
- Simon Provost ()

### Abstract

We present the Time-penalized Trees (TpT) splitter as exposed in
scikit-longitudinal. TpT extends decision-tree split selection to
longitudinal settings by penalizing candidate splits that rely on later
observation waves. This note focuses on software usage and
reproducibility rather than a full methodological treatment. We document
the current TpT API in scikit-longitudinal and provide two compact
real-data examples: a subject-level classification task on the primary
biliary cirrhosis follow-up data and a subject-level regression task on
the dietox pig growth data. The accompanying repository contains a
frozen Python environment, data preparation scripts, and Quarto sources
for the paper.

### Repository layout

- `paper.qmd`: Quarto source for the article.
- `environment.yml`: frozen Python environment used for the examples.
- `scripts/prepare_example_data.py`: downloads and preprocesses the open
  datasets.
- `analysis/tpt_examples.py`: helper functions used by the paper to run
  the two TpT examples.
- `data/processed/`: processed subject-level longitudinal datasets used
  in the paper.

### Reproducibility

1.  Create the environment described in `environment.yml`.
2.  Run `python scripts/prepare_example_data.py`.
3.  Ensure Quarto is available locally. The repository vendors the
    Computo extension under `_extensions/`.
4.  Render the HTML article with
    `quarto render paper.qmd --to computo-html`, or use
    `bash scripts/render_local.sh computo-html`.
5.  Render the PDF article with `quarto render paper.qmd --to
    computo-pdf`, or use `bash scripts/render_local.sh computo-pdf`.
