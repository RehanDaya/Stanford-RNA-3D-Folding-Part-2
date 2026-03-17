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


def parse_stoichiometry(stoichiometry: str) -> set[str]:
    """
    Parse stoichiometry string like 'B:1;A:2' into a set of author chain IDs {'A','B'}.
    """
    if not isinstance(stoichiometry, str) or not stoichiometry:
        return set()
    chains: set[str] = set()
    for part in stoichiometry.split(";"):
        part = part.strip()
        if not part:
            continue
        chain = part.split(":", 1)[0].strip()
        if chain:
            chains.add(chain)
    return chains


def _raw_target_metadata_map(split: Split) -> tuple[dict[str, pd.Series], list[str]]:
    """
    Build an unnormalized mapping target_id -> aggregated metadata row.
    Uses stoichiometry (author chain IDs) to select the relevant chains in rna_metadata.csv.
    """
    seq_df = load_sequences(split)
    meta = load_rna_metadata()

    cols = [
        "composition_rna_fraction",
        "total_structuredness_adjusted",
        "fraction_observed",
        "length",
        "resolution",
    ]
    keep_cols = [c for c in cols if c in meta.columns]

    out: dict[str, pd.Series] = {}
    for _, row in seq_df.iterrows():
        tid = str(row["target_id"])
        chains = parse_stoichiometry(row.get("stoichiometry", ""))
        m = meta[meta["pdb_id"].astype(str) == tid]
        if chains and "auth_chain_id" in m.columns:
            m2 = m[m["auth_chain_id"].astype(str).isin(chains)]
            if len(m2) > 0:
                m = m2
        if len(m) == 0:
            out[tid] = pd.Series({c: 0.0 for c in keep_cols})
            continue
        agg = m[keep_cols].mean(numeric_only=True)
        out[tid] = agg
    return out, keep_cols


def build_normalized_metadata_map(
    split: Split,
    normalize_using_split: Split = "train",
) -> dict[str, pd.Series]:
    """
    Build target_id -> metadata mapping, with per-feature standardization (z-score)
    computed from `normalize_using_split` targets.
    """
    raw_map, keep_cols = _raw_target_metadata_map(split)
    ref_map, _ = _raw_target_metadata_map(normalize_using_split)

    ref_df = pd.DataFrame.from_dict(ref_map, orient="index")[keep_cols]
    means = ref_df.mean(axis=0, numeric_only=True)
    stds = ref_df.std(axis=0, numeric_only=True).replace(0.0, 1.0)

    norm_map: dict[str, pd.Series] = {}
    for tid, s in raw_map.items():
        s2 = s.reindex(keep_cols).fillna(0.0)
        s_norm = (s2 - means) / stds
        norm_map[tid] = s_norm
    return norm_map


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

