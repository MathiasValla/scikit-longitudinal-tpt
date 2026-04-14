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

- `dietox_regression_long.csv`
  - Source dataset: `dietox` from the `geepack` R package.
  - Public source used for regeneration:
    `https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv`
  - Processing:
    - keep `Pig`, `Time`, `Weight`
    - define `final_weight` as the last recorded weight for each pig
    - remove the final visit from the longitudinal covariate history
    - define `duration` as the last remaining observation time
    - sort by pig id and observation time

The processed files can be regenerated with:

```bash
python scripts/prepare_example_data.py
```
