from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from data_io import DATA_DIR, add_target_and_resid, build_normalized_metadata_map, get_sequence_map, load_sample_submission
from features import build_per_residue_features
from model import BiLSTMCoordinateModel, ModelConfig


def load_model(checkpoint_path: Path, device: str | torch.device) -> BiLSTMCoordinateModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["config"])
    model = BiLSTMCoordinateModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_for_target(
    model: BiLSTMCoordinateModel,
    device: str | torch.device,
    target_id: str,
    sequence: str,
    meta_map: dict[str, "pd.Series"] | None = None,
) -> np.ndarray:
    """
    Run the model on a single target and return coordinates.

    Returns:
        coords: [L, num_structures, 3] numpy array.
    """
    metadata_row = None
    if meta_map is not None:
        metadata_row = meta_map.get(target_id)
    feats = build_per_residue_features(target_id, sequence, metadata_row=metadata_row)
    feats_t = torch.from_numpy(feats)[None, ...].to(device)  # [1,L,D]
    lengths = torch.tensor([feats.shape[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(feats_t, lengths)  # [1,L,N,3]
    coords = pred.squeeze(0).cpu().numpy()
    return coords


def build_submission(
    checkpoint_path: Path,
    output_path: Path,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Load a trained model checkpoint and generate submission.csv.
    """
    model = load_model(checkpoint_path, device)
    sample_sub = load_sample_submission()

    # Sequence lookup for test split.
    seq_map = get_sequence_map("test")
    meta_map = build_normalized_metadata_map("test", normalize_using_split="train")

    # We will accumulate predictions per target_id once, then fill rows.
    cache: dict[str, np.ndarray] = {}

    # Ensure coordinate columns are float so we can write model outputs.
    coord_cols: list[str] = []
    for i in range(1, 6):
        coord_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    sample_sub[coord_cols] = sample_sub[coord_cols].astype("float32")

    for idx, row in sample_sub.iterrows():
        target_id, resid = row["ID"].split("_", 1)
        if target_id not in cache:
            sequence = seq_map[target_id]
            coords = predict_for_target(model, device, target_id, sequence, meta_map=meta_map)
            cache[target_id] = coords

        coords = cache[target_id]
        resid_idx = int(resid) - 1
        if resid_idx < 0 or resid_idx >= coords.shape[0]:
            # Out-of-range residue index; leave zeros.
            continue
        per_res = coords[resid_idx]  # [N,3]

        # Flatten into x_1,y_1,z_1,... ordering.
        flat: list[float] = []
        for n in range(per_res.shape[0]):
            x, y, z = per_res[n]
            # Clip to allowed range.
            for v in (x, y, z):
                flat.append(float(np.clip(v, -999.999, 9999.999)))

        # Fill x_1,y_1,z_1,...,x_5,y_5,z_5 in the correct order.
        for col, val in zip(coord_cols, flat):
            sample_sub.at[idx, col] = val

    sample_sub.to_csv(output_path, index=False)
    print(f"Wrote submission to {output_path}")


if __name__ == "__main__":
    ckpt = Path("checkpoints") / "model_epoch5.pt"
    out = DATA_DIR / "submission.csv"
    build_submission(ckpt, out)

