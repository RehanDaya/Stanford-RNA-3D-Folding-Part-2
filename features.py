from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def load_msa_records(target_id: str) -> list[tuple[str, str]]:
    """
    Load MSA records as (record_id, aligned_sequence) pairs.
    If MSA is missing, returns an empty list.
    """
    msa_path = DATA_DIR / "MSA" / f"{target_id}.MSA.fasta"
    if not msa_path.exists():
        return []
    records = list(SeqIO.parse(str(msa_path), "fasta"))
    return [(rec.id, str(rec.seq).upper()) for rec in records]


def msa_query_mapped_stats(
    msa_records: list[tuple[str, str]],
    query_sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-residue MSA profile features by mapping alignment columns to query residues.

    Strategy:
      - Find the aligned query row (record id == 'query' preferred; else use first row).
      - For each alignment column j where query_aln[j] != '-', map it to residue index i
        in the ungapped query.
      - Compute A/C/G/U frequencies and Shannon entropy among homolog sequences at column j,
        ignoring gaps.

    Returns:
      freqs: [L_query, 4]
      entropy: [L_query, 1]

    If mapping fails, returns zero arrays.
    """
    Lq = len(query_sequence)
    if not msa_records:
        return np.zeros((Lq, 4), dtype=np.float32), np.zeros((Lq, 1), dtype=np.float32)

    # Pick query aligned row.
    query_aln = None
    for rid, seq in msa_records:
        if rid.lower() == "query":
            query_aln = seq
            break
    if query_aln is None:
        query_aln = msa_records[0][1]

    aln_len = len(query_aln)
    if aln_len == 0:
        return np.zeros((Lq, 4), dtype=np.float32), np.zeros((Lq, 1), dtype=np.float32)

    # Sanity: ensure all rows have same alignment length; if not, fallback to zeros.
    for _, s in msa_records:
        if len(s) != aln_len:
            return np.zeros((Lq, 4), dtype=np.float32), np.zeros((Lq, 1), dtype=np.float32)

    # Map alignment columns to residue indices in ungapped query.
    ungapped = query_aln.replace("-", "")
    if ungapped != query_sequence:
        # If mismatch, avoid introducing shifted/noisy features.
        return np.zeros((Lq, 4), dtype=np.float32), np.zeros((Lq, 1), dtype=np.float32)

    freqs = np.zeros((Lq, 4), dtype=np.float32)
    entropy = np.zeros((Lq, 1), dtype=np.float32)

    res_i = 0
    for j in range(aln_len):
        if query_aln[j] == "-":
            continue

        counts = np.zeros(4, dtype=np.float32)
        total = 0.0
        for _, s in msa_records:
            ch = s[j]
            if ch == "-":
                continue
            idx = NUC_TO_IDX.get(ch)
            if idx is not None:
                counts[idx] += 1.0
                total += 1.0

        if total > 0:
            p = counts / total
            freqs[res_i] = p
            mask = p > 0
            entropy[res_i, 0] = -np.sum(p[mask] * np.log2(p[mask]))

        res_i += 1
        if res_i >= Lq:
            break

    return freqs, entropy


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

    # MSA-based features (mapped to query residues using query aligned row when possible)
    msa_records = load_msa_records(target_id)
    msa_freqs, msa_entropy = msa_query_mapped_stats(msa_records, sequence)

    feature_blocks = [one_hot, pos, msa_freqs, msa_entropy]

    # Metadata scalars: always append a fixed-width vector so feature dim is consistent.
    meta_keys = [
        "composition_rna_fraction",
        "total_structuredness_adjusted",
        "fraction_observed",
        "length",
        "resolution",
    ]
    meta_vals: list[float] = []
    for k in meta_keys:
        if metadata_row is not None and k in metadata_row and pd.notna(metadata_row[k]):
            meta_vals.append(float(metadata_row[k]))
        else:
            meta_vals.append(0.0)
    meta_vec = np.array(meta_vals, dtype=np.float32)[None, :]
    meta_block = np.repeat(meta_vec, len(sequence), axis=0)
    feature_blocks.append(meta_block)

    return np.concatenate(feature_blocks, axis=1).astype(np.float32)

