from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    num_structures: int = 5  # number of alternative 3D predictions per residue


class BiLSTMCoordinateModel(nn.Module):
    """
    Lightweight sequence model that predicts C1' coordinates for each residue.

    Given per-residue feature vectors [B, L, D], it outputs coordinates of shape
    [B, L, num_structures, 3].
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(2 * cfg.hidden_dim, cfg.num_structures * 3)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] padded sequence of features.
            lengths: [B] original lengths for each sequence.

        Returns:
            coords: [B, L, num_structures, 3] padded predictions.
        """
        # Pack sequences for efficient LSTM over variable lengths.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        logits = self.out(out)  # [B, L, num_structures*3]
        B, L, _ = logits.shape
        coords = logits.view(B, L, self.cfg.num_structures, 3)
        return coords


def coordinate_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute loss between predicted and true coordinates.

    Args:
        pred_coords: [B, L, num_structures, 3]
        true_coords: [B, L, 3] (experimental structure for one conformation)
        lengths: [B] true lengths.

    We use the best-of-N structure loss: for each residue, pick the closest of
    the num_structures predictions.
    """
    # Expand true coords to [B, L, 1, 3] for broadcasting.
    true_exp = true_coords.unsqueeze(2)  # [B, L, 1, 3]
    diff = pred_coords - true_exp  # [B, L, N, 3]

    # Clamp differences to avoid overflow when squaring.
    diff = torch.clamp(diff, -1e4, 1e4)

    sq = (diff**2).sum(dim=-1)  # [B, L, N]
    best_sq, _ = sq.min(dim=-1)  # [B, L]

    mask = torch.arange(pred_coords.size(1), device=lengths.device)[None, :] < lengths[:, None]
    best_sq = best_sq * mask
    denom = mask.sum()
    return best_sq.sum() / torch.clamp(denom, min=1.0)

