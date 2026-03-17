from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO

from data_io import DATA_DIR


NUCLEOTIDES = ["A", "C", "G", "U"]
NUC_TO_IDX = {n: i for i, n in enumerate(NUCLEOTIDES)}


def one_hot_sequence(seq: str) -> np.ndarray:
    """
    One-hot encode an RNA sequence (A, C, G, U) into shape [L, 4].
    Non-ACGU characters are encoded as all zeros.
    """
    L = len(seq)
    arr = np.zeros((L, len(NUCLEOTIDES)), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = NUC_TO_IDX.get(ch.upper())
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


def positional_features(length: int) -> np.ndarray:
    """
    Simple positional features: relative position in [0,1].
    Shape: [L, 1]
    """
    if length <= 1:
        return np.zeros((length, 1), dtype=np.float32)
    positions = np.arange(length, dtype=np.float32) / float(length - 1)
    return positions[:, None]


def load_msa(target_id: str) -> List[str]:
    """
    Load MSA sequences for a given target from data/MSA/{target_id}.MSA.fasta.
    Returns a list of aligned sequences (strings with gaps '-').
    If MSA is missing, returns an empty list.
    """
    msa_path = DATA_DIR / "MSA" / f"{target_id}.MSA.fasta"
    if not msa_path.exists():
        return []
    records = list(SeqIO.parse(str(msa_path), "fasta"))
    return [str(rec.seq).upper() for rec in records]


def msa_position_stats(msa_seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-position nucleotide frequencies and Shannon entropy from an aligned MSA.
    Returns:
      freqs: [L, 4] frequencies for A,C,G,U ignoring gaps.
      entropy: [L, 1] Shannon entropy in bits.
    If msa_seqs is empty, returns arrays of zeros with length 0.
    """
    if not msa_seqs:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    L = len(msa_seqs[0])
    freqs = np.zeros((L, 4), dtype=np.float32)
    entropy = np.zeros((L, 1), dtype=np.float32)

    for pos in range(L):
        counts = np.zeros(4, dtype=np.float32)
        total = 0.0
        for s in msa_seqs:
            ch = s[pos]
            if ch == "-":
                continue
            idx = NUC_TO_IDX.get(ch)
            if idx is not None:
                counts[idx] += 1.0
                total += 1.0
        if total > 0:
            p = counts / total
            freqs[pos] = p
            mask = p > 0
            entropy[pos, 0] = -np.sum(p[mask] * np.log2(p[mask]))

    return freqs, entropy


def build_per_residue_features(
    target_id: str,
    sequence: str,
    metadata_row: pd.Series | None = None,
) -> np.ndarray:
    """
    Build per-residue feature matrix for a target.

    Components:
      - one-hot nucleotide encoding [L,4]
      - positional features [L,1]
      - optional metadata scalars broadcast per residue (length, composition, etc.)
      - optional MSA-derived frequency + entropy features [L,5]
    """
    one_hot = one_hot_sequence(sequence)
    pos = positional_features(len(sequence))

    # MSA-based features (if available)
    msa = load_msa(target_id)
    msa_freqs, msa_entropy = msa_position_stats(msa)

    # If MSA length differs (due to gaps at ends), we will truncate/pad to match sequence length.
    def match_length(arr: np.ndarray, L: int) -> np.ndarray:
        if arr.shape[0] == 0:
            return np.zeros((L, arr.shape[1]), dtype=np.float32)
        if arr.shape[0] == L:
            return arr
        if arr.shape[0] > L:
            return arr[:L]
        pad = np.zeros((L - arr.shape[0], arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    msa_freqs = match_length(msa_freqs, len(sequence))
    msa_entropy = match_length(msa_entropy, len(sequence))

    feature_blocks = [one_hot, pos, msa_freqs, msa_entropy]

    # Metadata scalars: choose a few useful columns if provided.
    if metadata_row is not None:
        meta_cols = []
        for col in [
            "composition_rna_fraction",
            "total_structuredness_adjusted",
            "length",
            "resolution",
        ]:
            if col in metadata_row and pd.notna(metadata_row[col]):
                meta_cols.append(float(metadata_row[col]))
        if meta_cols:
            meta_vec = np.array(meta_cols, dtype=np.float32)[None, :]
            meta_block = np.repeat(meta_vec, len(sequence), axis=0)
            feature_blocks.append(meta_block)

    return np.concatenate(feature_blocks, axis=1).astype(np.float32)

