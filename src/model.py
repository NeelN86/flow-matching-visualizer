"""Velocity MLP: the core model trained by flow matching.

The network learns v_theta(x, t) such that integrating dx/dt = v_theta(x, t)
from t=0 (noise) to t=1 transports samples toward the target distribution.

Architecture: Linear(3→256) → SiLU → Linear(256→256) → SiLU ×2 → Linear(256→2)
Input:  (x, y, t) — spatial position concatenated with the current time step.
Output: (vx, vy)  — predicted velocity at that position and time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VelocityMLP(nn.Module):
    def __init__(self, hidden_width: int = 256, n_hidden: int = 3) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = 3  # (x, y, t)
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_width), nn.SiLU()]
            in_dim = hidden_width
        layers.append(nn.Linear(hidden_width, 2))  # → (vx, vy)

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict velocity at position x and time t.

        Args:
            x: [B, 2]  spatial coordinates.
            t: [B]     times in [0, 1].

        Returns:
            [B, 2] predicted velocity (vx, vy).
        """
        # Concatenate position and time so the network is explicitly
        # conditioned on t — critical for learning a time-varying field.
        t_col = t.view(-1, 1)          # [B, 1]
        inp = torch.cat([x, t_col], dim=1)  # [B, 3]
        return self.net(inp)
