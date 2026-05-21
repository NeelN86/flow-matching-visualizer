"""Velocity MLP: the core model trained by flow matching.

The network learns v_theta(x, t) such that integrating dx/dt = v_theta(x, t)
from t=0 (noise) to t=1 transports samples toward the target distribution.

Unconditioned (default):
  Input:  (x, y, t)                    — 3-dim
  Output: (vx, vy)

Class-conditioned (n_classes set):
  Input:  (x, y, t, class_emb[0..7])  — 11-dim
  Output: (vx, vy)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VelocityMLP(nn.Module):
    def __init__(
        self,
        hidden_width: int = 256,
        n_hidden: int = 3,
        n_classes: int | None = None,
        emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim = emb_dim

        if n_classes is not None:
            self.class_emb = nn.Embedding(n_classes, emb_dim)
            in_dim = 3 + emb_dim   # (x, y, t, class_emb...)
        else:
            self.class_emb = None  # type: ignore[assignment]
            in_dim = 3             # (x, y, t)

        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_width), nn.SiLU()]
            in_dim = hidden_width
        layers.append(nn.Linear(hidden_width, 2))  # → (vx, vy)

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        """Predict velocity at position x and time t.

        Args:
            x: [B, 2]  spatial coordinates.
            t: [B]     times in [0, 1].
            c: [B]     integer class labels (required if n_classes was set).

        Returns:
            [B, 2] predicted velocity (vx, vy).
        """
        t_col = t.view(-1, 1)
        inp = torch.cat([x, t_col], dim=1)  # [B, 3]

        if self.class_emb is not None:
            if c is None:
                raise ValueError("Class label 'c' required for class-conditioned model.")
            inp = torch.cat([inp, self.class_emb(c)], dim=1)  # [B, 3+emb_dim]

        return self.net(inp)
