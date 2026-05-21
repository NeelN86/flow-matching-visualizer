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

For class-conditioned training (class_label set), x1 is sampled from a
pre-built MNIST latent pool and the class index is fed as an embedding.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from src.data import TARGETS, sample_source
from src.interpolation import interpolate, target_velocity
from src.model import VelocityMLP


def train(
    target_name: str = "two_moons",
    steps: int = 5_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    progress_cb: Callable[[int, float], None] | None = None,
    # --- class-conditioned MNIST flow matching ---
    class_label: int | None = None,
    mnist_sampler: Callable[[int], torch.Tensor] | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Train a VelocityMLP via flow matching.

    Args:
        target_name:    Key into TARGETS registry. Ignored when class_label is set.
        steps:          Number of gradient steps.
        batch_size:     Samples drawn per step.
        lr:             Adam learning rate.
        progress_cb:    Optional callback(step, loss) called every step.
        class_label:    Digit class (0-9) for class-conditioned MNIST flow matching.
                        When set, mnist_sampler must also be provided.
        mnist_sampler:  Pre-built sampler from make_mnist_class_sampler(); required
                        when class_label is not None.

    Returns:
        (model, loss_history) where loss_history[i] is the scalar loss at step i.
    """
    conditioned = class_label is not None

    if conditioned:
        if mnist_sampler is None:
            raise ValueError("mnist_sampler is required when class_label is set.")
        target_sampler = mnist_sampler
        model = VelocityMLP(n_classes=10)
        c_tensor = torch.full((batch_size,), class_label, dtype=torch.long)
    else:
        if target_name not in TARGETS:
            raise ValueError(f"Unknown target '{target_name}'. Choose from {list(TARGETS)}")
        target_sampler = TARGETS[target_name]
        model = VelocityMLP()
        c_tensor = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    model.train()
    for step in range(steps):
        x0 = sample_source(batch_size)
        x1 = target_sampler(batch_size)
        t  = torch.rand(batch_size)

        x_t      = interpolate(x0, x1, t)
        v_target = target_velocity(x0, x1)
        v_pred   = model(x_t, t, c_tensor)
        loss     = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scalar = loss.item()
        loss_history.append(scalar)

        if progress_cb is not None:
            progress_cb(step, scalar)

    return model, loss_history
