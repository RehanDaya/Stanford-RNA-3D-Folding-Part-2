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

    # Hyperparameters for loss terms.
    # - dist_window controls the sequence-distance range for distance-matrix supervision.
    dist_window = 16
    w_dist = 0.1

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

    def center_and_scale(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Center and scale coordinates.
        Returns:
          Xn: normalized coords
          mu: centroid
          s: scale (RMS radius)
        """
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        s = torch.sqrt(torch.mean(torch.sum(Xc * Xc, dim=-1))).clamp(min=1e-6)
        Xn = Xc / s
        return Xn, mu, s

    def local_distance_loss(P_full: torch.Tensor, Q_full: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Compare local (sequence-neighbor) distance patterns for offsets 1..dist_window.
        P_full and Q_full are [L,3] in the same (normalized) coordinate frame.
        valid is [L] indicating which residues are present.
        """
        losses = []
        Lloc = P_full.shape[0]
        w = min(dist_window, Lloc - 1)
        if w <= 0:
            return torch.tensor(0.0, device=P_full.device)
        for off in range(1, w + 1):
            m = valid[:-off] & valid[off:]
            if not torch.any(m):
                continue
            dP = torch.linalg.norm(P_full[off:][m] - P_full[:-off][m], dim=-1)
            dQ = torch.linalg.norm(Q_full[off:][m] - Q_full[:-off][m], dim=-1)
            losses.append(torch.mean((dP - dQ) ** 2))
        if not losses:
            return torch.tensor(0.0, device=P_full.device)
        return torch.stack(losses).mean()

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
            Qn, _, _ = center_and_scale(Qv)

            # For this conformation, best over predicted structures.
            best_k = None
            for n in range(N):
                P = pred_b[:, n, :]  # [L,3]
                Pv = P[valid]
                Pn, _, _ = center_and_scale(Pv)

                # Clamp to avoid numerical overflow.
                Pn = torch.clamp(Pn, -1e4, 1e4)
                Qn = torch.clamp(Qn, -1e4, 1e4)

                Palign = kabsch_align(Pn, Qn)
                mse = torch.mean((Palign - Qn) ** 2)

                # Distance-matrix supervision on a local window, using normalized coordinates.
                # Use aligned predicted coords in the full-length indexing for consistency.
                # Reconstruct a full-length normalized-aligned P by placing the aligned valid
                # residues back into their positions; invalid positions remain unused by mask.
                P_full = torch.zeros((maxL, 3), device=device, dtype=pred_coords.dtype)
                Q_full = torch.zeros((maxL, 3), device=device, dtype=pred_coords.dtype)
                P_full[valid] = Palign
                Q_full[valid] = Qn
                dist_loss = local_distance_loss(P_full, Q_full, valid)

                total = mse + w_dist * dist_loss
                if best_k is None or total < best_k:
                    best_k = total
            if best_k is None:
                continue
            if best_b is None or best_k < best_b:
                best_b = best_k

        if best_b is None:
            # No valid conformation; return 0 for this sample.
            best_b = torch.tensor(0.0, device=device)
        losses.append(best_b)

    return torch.stack(losses).mean()

