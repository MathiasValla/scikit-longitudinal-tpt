# Data provenance

The current article uses only `raw/pbcseq.csv`, a public snapshot of
`pbcseq` from the R package `survival`. Its public CSV source is
[Rdatasets](https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv).
`analysis/tpt_case_study.py` reads the bundled CSV, excludes patients
whose outcome in the five-to-ten-year window is unknown, and builds
features exclusively from visits through year five. No data download or
private inputs are required to run this analysis.

The dataset documentation is available in the
[R survival manual](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbcseq.html).
