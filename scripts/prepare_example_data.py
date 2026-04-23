"""Download and preprocess the open longitudinal datasets used in the paper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PBCSEQ_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv"
PRODUC_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Produc.csv"

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


def prepare_produc() -> pd.DataFrame:
    """Prepare the state-level regression dataset from Produc."""

    raw = _download_if_missing(PRODUC_URL, RAW_DIR / "produc.csv")
    base_year = int(raw["year"].min())

    processed = raw[["state", "year", "pcap", "pc", "emp", "unemp", "gsp"]].copy()
    processed = processed.sort_values(["state", "year"], kind="mergesort").reset_index(
        drop=True
    )
    last_year = processed.groupby("state")["year"].transform("max")
    processed["final_gsp"] = processed.groupby("state")["gsp"].transform("last")
    processed = processed[processed["year"] < last_year].copy()
    processed["time_index"] = processed["year"] - base_year
    processed["duration"] = processed.groupby("state")["time_index"].transform("max")
    processed = processed.sort_values(["state", "time_index"], kind="mergesort").reset_index(drop=True)
    processed.to_csv(PROCESSED_DIR / "produc_regression_long.csv", index=False)
    return processed


def main() -> None:
    """Regenerate all processed datasets used in the paper."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pbcseq = prepare_pbcseq()
    produc = prepare_produc()

    print(
        "Prepared pbcseq classification data:",
        {
            "rows": int(pbcseq.shape[0]),
            "subjects": int(pbcseq["id"].nunique()),
            "event_subjects": int(pbcseq.groupby("id")["event"].first().sum()),
        },
    )
    print(
        "Prepared Produc regression data:",
        {
            "rows": int(produc.shape[0]),
            "states": int(produc["state"].nunique()),
            "max_time_index": int(produc["time_index"].max()),
            "predictors": ["pcap", "pc", "emp", "unemp"],
        },
    )


if __name__ == "__main__":
    main()
