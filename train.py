from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_io import DATA_DIR, add_target_and_resid, build_normalized_metadata_map, get_sequence_map, load_labels
from features import build_per_residue_features
from model import BiLSTMCoordinateModel, ModelConfig, coordinate_loss


class RNATargetDataset(Dataset):
    """
    Dataset that yields per-target feature and label tensors.
    Each item corresponds to one target_id.
    """

    def __init__(self, split: str, max_targets: int | None = None):
        assert split in {"train", "validation"}
        self.split = split

        print(f"Loading {split}_labels.parquet ...")
        labels_df = load_labels(split)
        labels_df = add_target_and_resid(labels_df)
        self.seq_map = get_sequence_map(split)
        self.meta_map = build_normalized_metadata_map(split, normalize_using_split="train")

        # Identify how many coordinate conformations exist in this split (train has 1; validation has up to 40).
        coord_indices: list[int] = []
        for col in labels_df.columns:
            if col.startswith("x_"):
                try:
                    coord_indices.append(int(col.split("_", 1)[1]))
                except ValueError:
                    pass
        coord_indices = sorted(set(coord_indices))
        if not coord_indices:
            raise ValueError("No coordinate columns (x_*) found in labels.")

        self.num_confs = len(coord_indices)
        x_cols = [f"x_{i}" for i in coord_indices]
        y_cols = [f"y_{i}" for i in coord_indices]
        z_cols = [f"z_{i}" for i in coord_indices]

        # Group labels by target_id and sort by resid (fast path via groupby).
        print(f"Grouping {split} labels by target_id ...")
        groups: Dict[str, np.ndarray] = {}
        for tid, g in tqdm(
            labels_df.groupby("target_id"), desc=f"Building {split} target groups"
        ):
            g = g.sort_values("resid")
            coords = np.stack(
                [
                    g[x_cols].to_numpy(dtype=np.float32),
                    g[y_cols].to_numpy(dtype=np.float32),
                    g[z_cols].to_numpy(dtype=np.float32),
                ],
                axis=-1,
            )  # [L, K, 3]
            # Treat sentinel missing values (e.g. -1e18) as NaN so loss can ignore them.
            coords[coords < -1e17] = np.nan
            groups[tid] = coords

        self.targets: List[str] = sorted(groups.keys())
        if max_targets is not None:
            self.targets = self.targets[:max_targets]
        print(f"{split} split: using {len(self.targets)} targets")
        self.groups: Dict[str, np.ndarray] = {tid: groups[tid] for tid in self.targets}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        tid = self.targets[idx]
        seq = self.seq_map[tid]
        coords = self.groups[tid]
        meta_row = self.meta_map.get(tid)
        feats = build_per_residue_features(tid, seq, metadata_row=meta_row)
        # Truncate/pad coords to match sequence length if needed.
        L = len(seq)
        if coords.shape[0] > L:
            coords = coords[:L]
        elif coords.shape[0] < L:
            pad = np.full((L - coords.shape[0], coords.shape[1], 3), np.nan, dtype=np.float32)
            coords = np.concatenate([coords, pad], axis=0)
        return tid, feats, coords


def collate_batch(
    batch: List[Tuple[str, np.ndarray, np.ndarray]]
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a batch of variable-length targets into padded tensors.
    Returns:
      target_ids, feats [B,L,D], coords [B,L,3], lengths [B]
    """
    target_ids = [b[0] for b in batch]
    # Copy NumPy arrays before converting to tensors to avoid non-writable warnings.
    feats_list = [torch.from_numpy(b[1].copy()) for b in batch]
    coords_list = [torch.from_numpy(b[2].copy()) for b in batch]
    lengths = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
    max_len = lengths.max().item()
    D = feats_list[0].shape[1]

    B = len(batch)
    feats = torch.zeros((B, max_len, D), dtype=torch.float32)
    K = coords_list[0].shape[1]
    coords = torch.full((B, max_len, K, 3), float("nan"), dtype=torch.float32)
    for i, (f, c) in enumerate(zip(feats_list, coords_list)):
        L = f.shape[0]
        feats[i, :L] = f
        coords[i, :L] = c

    # Sanitize any NaN/Inf values to keep losses finite.
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return target_ids, feats, coords, lengths


def train_model(
    output_dir: Path,
    num_epochs: int = 5,
    batch_size: int = 100,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Train the BiLSTMCoordinateModel on train split and evaluate on validation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building datasets ...")
    # Limit targets for faster experimentation; increase/remove limits for full runs.
    train_ds = RNATargetDataset("train", max_targets=200)
    val_ds = RNATargetDataset("validation", max_targets=50)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Infer input_dim from one batch.
    _, feats_sample, _, lengths_sample = next(iter(train_loader))
    input_dim = feats_sample.shape[-1]

    cfg = ModelConfig(input_dim=input_dim)
    model = BiLSTMCoordinateModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        print(f"\nStarting epoch {epoch}/{num_epochs} ...")
        model.train()
        train_loss = 0.0
        n_batches = 0
        for _, feats, coords, lengths in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            feats = feats.to(device)
            coords = coords.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            pred = model(feats, lengths)
            loss = coordinate_loss(pred, coords, lengths)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Encountered non-finite loss in training batch; skipping this batch.")
                continue
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for _, feats, coords, lengths in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                feats = feats.to(device)
                coords = coords.to(device)
                lengths = lengths.to(device)
                pred = model(feats, lengths)
                loss = coordinate_loss(pred, coords, lengths)
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Encountered non-finite loss in validation batch; skipping this batch.")
                    continue
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        ckpt_path = output_dir / f"model_epoch{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            ckpt_path,
        )


if __name__ == "__main__":
    train_model(output_dir=Path("checkpoints"))

