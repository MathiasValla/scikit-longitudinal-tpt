"""Download and preprocess the open longitudinal datasets used in the paper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PBCSEQ_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv"
DIETOX_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _download_if_missing(url: str, destination: Path) -> pd.DataFrame:
    """Download a CSV if it is not already present on disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return pd.read_csv(destination)

    frame = pd.read_csv(url)
    frame.to_csv(destination, index=False)
    return frame


def prepare_pbcseq() -> pd.DataFrame:
    """Prepare the subject-level classification dataset from pbcseq."""

    raw = _download_if_missing(PBCSEQ_URL, RAW_DIR / "pbcseq.csv")
    processed = raw[
        ["id", "day", "futime", "status", "bili", "albumin", "protime", "ast"]
    ].copy()
    processed["event"] = (processed["status"] != 0).astype(int)
    processed = processed.drop(columns=["status"])
    processed = processed.sort_values(["id", "day"], kind="mergesort").reset_index(
        drop=True
    )
    processed.to_csv(PROCESSED_DIR / "pbcseq_classification_long.csv", index=False)
    return processed


def prepare_dietox() -> pd.DataFrame:
    """Prepare the subject-level regression dataset from dietox."""

    raw = _download_if_missing(DIETOX_URL, RAW_DIR / "dietox.csv")
    processed = raw[["Pig", "Time", "Weight"]].copy()
    last_time = processed.groupby("Pig")["Time"].transform("max")
    processed["final_weight"] = processed.groupby("Pig")["Weight"].transform("last")
    processed = processed[processed["Time"] < last_time].copy()
    processed["duration"] = processed.groupby("Pig")["Time"].transform("max")
    processed = processed.sort_values(["Pig", "Time"], kind="mergesort").reset_index(
        drop=True
    )
    processed.to_csv(PROCESSED_DIR / "dietox_regression_long.csv", index=False)
    return processed


def main() -> None:
    """Regenerate all processed datasets used in the paper."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pbcseq = prepare_pbcseq()
    dietox = prepare_dietox()

    print(
        "Prepared pbcseq classification data:",
        {
            "rows": int(pbcseq.shape[0]),
            "subjects": int(pbcseq["id"].nunique()),
            "event_subjects": int(pbcseq.groupby("id")["event"].first().sum()),
        },
    )
    print(
        "Prepared dietox regression data:",
        {
            "rows": int(dietox.shape[0]),
            "subjects": int(dietox["Pig"].nunique()),
            "max_time": int(dietox["Time"].max()),
        },
    )


if __name__ == "__main__":
    main()
