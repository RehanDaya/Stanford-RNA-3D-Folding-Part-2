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
        self.scale_head = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Softplus(),
        )

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
        # Predict a positive per-target scale and apply it to all coordinates.
        # This helps the model learn a consistent coordinate scale without relying on
        # absolute coordinate magnitudes early in training.
        mask = torch.arange(L, device=lengths.device)[None, :] < lengths[:, None]  # [B,L]
        mask_f = mask.to(out.dtype).unsqueeze(-1)  # [B,L,1]
        denom = torch.clamp(mask_f.sum(dim=1), min=1.0)  # [B,1]
        pooled = (out * mask_f).sum(dim=1) / denom  # [B,2H]
        scale = self.scale_head(pooled).view(B, 1, 1, 1)  # [B,1,1,1]
        coords = coords * scale
        return coords


def coordinate_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute loss between predicted and true coordinates.

    Args:
        pred_coords: [B, L, N, 3] predicted coordinates for N structures
        true_coords: [B, L, K, 3] experimental coordinates for K conformations (NaN where missing)
        lengths: [B] true lengths.

    Loss:
      - For each (batch item), for each true conformation k, we align each predicted
        structure n to that conformation using Kabsch (rigid rotation + translation),
        compute mean squared error on valid residues, then take min over n and k.
    """
    device = pred_coords.device
    B, L, N, _ = pred_coords.shape
    _, _, K, _ = true_coords.shape

    def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Align P to Q with Kabsch rotation. Both are [M,3] and assumed centered.
        Returns aligned P [M,3].
        """
        C = P.T @ Q
        V, S, Wt = torch.linalg.svd(C, full_matrices=False)
        d = torch.sign(torch.det(V @ Wt))
        D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
        R = V @ D @ Wt
        return P @ R

    losses = []
    for b in range(B):
        maxL = int(lengths[b].item())
        if maxL <= 0:
            losses.append(torch.tensor(0.0, device=device))
            continue

        pred_b = pred_coords[b, :maxL]  # [L, N, 3]
        true_b = true_coords[b, :maxL]  # [L, K, 3]

        best_b = None
        for k in range(K):
            Q = true_b[:, k, :]  # [L,3]
            valid = torch.isfinite(Q).all(dim=-1)  # [L]
            if valid.sum() < 3:
                continue
            Qv = Q[valid]
            Qc = Qv - Qv.mean(dim=0, keepdim=True)

            # For this conformation, best over predicted structures.
            best_k = None
            for n in range(N):
                P = pred_b[:, n, :]  # [L,3]
                Pv = P[valid]
                Pc = Pv - Pv.mean(dim=0, keepdim=True)

                # Clamp to avoid numerical overflow.
                Pc = torch.clamp(Pc, -1e4, 1e4)
                Qc2 = torch.clamp(Qc, -1e4, 1e4)

                Palign = kabsch_align(Pc, Qc2)
                mse = torch.mean((Palign - Qc2) ** 2)
                if best_k is None or mse < best_k:
                    best_k = mse
            if best_k is None:
                continue
            if best_b is None or best_k < best_b:
                best_b = best_k

        if best_b is None:
            # No valid conformation; return 0 for this sample.
            best_b = torch.tensor(0.0, device=device)
        losses.append(best_b)

    return torch.stack(losses).mean()

