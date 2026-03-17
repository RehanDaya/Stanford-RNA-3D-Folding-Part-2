from pathlib import Path
from typing import Literal

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


Split = Literal["train", "validation", "test"]


def load_sequences(split: Split) -> pd.DataFrame:
    """
    Load {split}_sequences.csv into a DataFrame.
    """
    fname = f"{split}_sequences.csv"
    path = DATA_DIR / fname
    return pd.read_csv(path, low_memory=False)


def load_labels(split: Literal["train", "validation"]) -> pd.DataFrame:
    """
    Load {split}_labels.csv into a DataFrame.
    """
    fname = f"{split}_labels.csv"
    path = DATA_DIR / fname
    return pd.read_csv(path, low_memory=False)


def load_sample_submission() -> pd.DataFrame:
    """
    Load sample_submission.csv into a DataFrame.
    """
    path = DATA_DIR / "sample_submission.csv"
    return pd.read_csv(path)


def parse_id_column(id_value: str) -> tuple[str, int]:
    """
    Split an ID like '8ZNQ_1' into (target_id, resid).
    """
    target_id, resid_str = id_value.split("_", 1)
    return target_id, int(resid_str)


def add_target_and_resid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with an ID column, add 'target_id' and 'resid' columns.
    """
    targets, resids = zip(*(parse_id_column(x) for x in df["ID"].astype(str)))
    df = df.copy()
    df["target_id"] = targets
    df["resid"] = resids
    return df


def load_rna_metadata() -> pd.DataFrame:
    """
    Load extra/rna_metadata.csv containing per-chain RNA metadata.
    """
    path = DATA_DIR / "extra" / "rna_metadata.csv"
    return pd.read_csv(path)


def get_sequence_map(split: Split) -> dict[str, str]:
    """
    Build a mapping from target_id -> RNA sequence string for a given split.
    """
    seq_df = load_sequences(split)
    return dict(zip(seq_df["target_id"].astype(str), seq_df["sequence"].astype(str)))


def get_metadata_for_targets(target_ids: list[str]) -> pd.DataFrame:
    """
    Filter RNA metadata rows for the given target_ids (matching on target_id column).
    """
    meta = load_rna_metadata()
    mask = meta["target_id"].isin(target_ids)
    return meta.loc[mask].copy()

