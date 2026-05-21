"""Training loop for the flow-matching velocity MLP.

At every step we:
  1. Draw a batch of source samples x0 ~ N(0, I).
  2. Draw a batch of target samples x1 from the chosen distribution.
  3. Draw a batch of times t ~ U[0, 1].
  4. Compute the flow-matching MSE loss:
       L = E[ || v_theta(x_t, t) - (x1 - x0) ||^2 ]
     where x_t = (1-t)*x0 + t*x1.
  5. Backpropagate and step Adam.

No data loader is needed — distributions are sampled on the fly each step,
so the effective dataset is infinite.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from src.data import TARGETS, sample_source
from src.interpolation import flow_matching_loss
from src.model import VelocityMLP


def train(
    target_name: str = "two_moons",
    steps: int = 5_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    progress_cb: Callable[[int, float], None] | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Train a VelocityMLP via flow matching.

    Args:
        target_name:  Key into TARGETS registry ("two_moons", "gaussian_mixture",
                      "checkerboard").
        steps:        Number of gradient steps.
        batch_size:   Samples drawn per step from both source and target.
        lr:           Adam learning rate.
        progress_cb:  Optional callback(step, loss) called every step. Useful
                      for streaming live loss to a Gradio component.

    Returns:
        (model, loss_history) where loss_history[i] is the scalar loss at step i.
    """
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target '{target_name}'. Choose from {list(TARGETS)}")

    target_sampler = TARGETS[target_name]
    model = VelocityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    model.train()
    for step in range(steps):
        x0 = sample_source(batch_size)              # noise samples
        x1 = target_sampler(batch_size)             # target samples
        t = torch.rand(batch_size)                  # t ~ U[0, 1]

        optimizer.zero_grad()
        loss = flow_matching_loss(model, x0, x1, t)
        loss.backward()
        optimizer.step()

        scalar = loss.item()
        loss_history.append(scalar)

        if progress_cb is not None:
            progress_cb(step, scalar)

    return model, loss_history
