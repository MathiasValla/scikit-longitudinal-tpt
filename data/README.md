# Data provenance

This repository uses two small public longitudinal datasets.

- `pbcseq_classification_long.csv`
  - Source dataset: `pbcseq` from the `survival` R package.
  - Public source used for regeneration:
    `https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv`
  - Processing:
    - keep the columns `id`, `day`, `futime`, `bili`, `albumin`, `protime`, `ast`
    - create a binary subject-level outcome `event = 1(status != 0)`
    - sort by subject id and follow-up day

- `produc_regression_long.csv`
  - Source dataset: `Produc` from the `plm` R package.
  - Public source used for regeneration:
    `https://vincentarelbundock.github.io/Rdatasets/csv/plm/Produc.csv`
  - Processing:
    - keep `state`, `year`, `pcap`, `pc`, `emp`, `unemp`, `gsp`
    - sort chronologically by `state` and `year`
    - define `final_gsp` as the last recorded gross state product for each state
    - remove the final year from the longitudinal covariate history
    - define `time_index` as `year - 1970`
    - define `duration` as the last remaining observed wave index
    - sort by state and wave index

The processed files can be regenerated with:

```bash
python scripts/prepare_example_data.py
```
