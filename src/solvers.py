"""ODE solvers for inference-time sample generation.

After training, we generate samples by integrating the learned velocity field
from t=0 (noise) to t=1 (target). The full trajectory is returned so the
visualizer can render particle trails frame-by-frame.

Euler: first-order, cheap, good enough for smooth learned fields.
RK4:   fourth-order, 4x more model evals per step but far more accurate
       on curved paths — useful for the solver-comparison stretch goal.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.model import VelocityMLP


@torch.no_grad()
def euler_integrate(
    model: VelocityMLP,
    x0: Tensor,
    n_steps: int = 100,
    t_span: tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """Euler integration of dx/dt = v_theta(x, t) from t_span[0] to t_span[1].

    Args:
        model:   Trained VelocityMLP.
        x0:      [B, 2] initial positions (noise samples at t=0).
        n_steps: Number of uniform Euler steps. More steps = smoother trails.
        t_span:  (t_start, t_end) — use (0, 1) for full transport.

    Returns:
        trajectory: [n_steps+1, B, 2] — x at every time step including x0.
    """
    model.eval()
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    B = x0.shape[0]

    x = x0.clone()
    trajectory = [x]

    for i in range(n_steps):
        t = t0 + i * dt
        t_batch = torch.full((B,), t, dtype=x.dtype)
        v = model(x, t_batch)
        x = x + dt * v          # Euler step: x_{k+1} = x_k + dt * v(x_k, t_k)
        trajectory.append(x)

    return torch.stack(trajectory, dim=0)  # [n_steps+1, B, 2]


@torch.no_grad()
def rk4_integrate(
    model: VelocityMLP,
    x0: Tensor,
    n_steps: int = 100,
    t_span: tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """4th-order Runge-Kutta integration of dx/dt = v_theta(x, t).

    Uses 4 model evaluations per step (k1..k4) to achieve O(dt^4) local error
    vs Euler's O(dt^2). Same signature as euler_integrate for easy comparison.

    Returns:
        trajectory: [n_steps+1, B, 2]
    """
    model.eval()
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    B = x0.shape[0]

    def _v(x: Tensor, t: float) -> Tensor:
        t_batch = torch.full((B,), t, dtype=x.dtype)
        return model(x, t_batch)

    x = x0.clone()
    trajectory = [x]

    for i in range(n_steps):
        t = t0 + i * dt
        k1 = _v(x,               t)
        k2 = _v(x + 0.5*dt*k1,   t + 0.5*dt)
        k3 = _v(x + 0.5*dt*k2,   t + 0.5*dt)
        k4 = _v(x + dt*k3,       t + dt)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(x)

    return torch.stack(trajectory, dim=0)  # [n_steps+1, B, 2]


@torch.no_grad()
def velocity_field_on_grid(
    model: VelocityMLP,
    t: float,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    res: int = 20,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the learned velocity field on a regular grid.

    Args:
        model:  Trained VelocityMLP.
        t:      Time in [0, 1] at which to evaluate the field.
        bounds: (x_min, x_max, y_min, y_max).
        res:    Grid resolution along each axis.

    Returns:
        (X, Y, U, V) — each [res, res] — ready for ax.quiver(X, Y, U, V).
    """
    model.eval()
    x_min, x_max, y_min, y_max = bounds
    xs = torch.linspace(x_min, x_max, res)
    ys = torch.linspace(y_min, y_max, res)

    # meshgrid returns (res, res) grids; 'ij' indexing makes X vary along axis-0
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [res*res, 2]

    t_batch = torch.full((grid.shape[0],), t)
    v = model(grid, t_batch)                   # [res*res, 2]

    U = v[:, 0].reshape(res, res)
    V = v[:, 1].reshape(res, res)
    return X, Y, U, V
