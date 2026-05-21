"""Linear interpolation and flow-matching loss.

Flow matching works by constructing straight-line paths between source samples
x0 and target samples x1:

    x(t) = (1 - t) * x0 + t * x1,  t in [0, 1]

The velocity along this path is constant:

    dx/dt = x1 - x0

We train the MLP to predict this velocity at arbitrary (x_t, t) pairs. At
inference time, integrating the learned velocity from t=0 to t=1 reproduces
the target distribution.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Linearly interpolate between x0 and x1 at time t.

    Args:
        x0: [B, 2] source samples (noise).
        x1: [B, 2] target samples.
        t:  [B]    times in [0, 1].

    Returns:
        x_t: [B, 2]  point along the straight path at time t.
    """
    t = t.view(-1, 1)  # broadcast over the 2 spatial dims
    return (1.0 - t) * x0 + t * x1


def target_velocity(x0: Tensor, x1: Tensor) -> Tensor:
    """The ground-truth velocity for the straight-line path from x0 to x1.

    Because x(t) = (1-t)*x0 + t*x1, differentiating gives dx/dt = x1 - x0
    — a constant that does not depend on t. This is what the MLP regresses.

    Args:
        x0: [B, 2] source samples.
        x1: [B, 2] target samples.

    Returns:
        [B, 2] velocity vectors.
    """
    return x1 - x0


def flow_matching_loss(
    model: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
) -> Tensor:
    """MSE between the model's predicted velocity and the ground-truth velocity.

    At each training step we:
      1. Compute x_t, the noisy interpolant at time t.
      2. Ask the model what velocity it predicts at (x_t, t).
      3. Penalise the L2 distance from the true velocity (x1 - x0).

    Args:
        model: callable (x_t [B,2], t [B]) -> v_pred [B,2].
        x0:    [B, 2] source samples.
        x1:    [B, 2] target samples.
        t:     [B]    uniformly sampled times.

    Returns:
        Scalar MSE loss.
    """
    x_t = interpolate(x0, x1, t)
    v_target = target_velocity(x0, x1)
    v_pred = model(x_t, t)
    return F.mse_loss(v_pred, v_target)
