"""Tests for src/solvers.py."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from src.model import VelocityMLP
from src.solvers import euler_integrate, rk4_integrate, velocity_field_on_grid


def _dummy_model() -> VelocityMLP:
    """Untrained model — used only for shape / API tests."""
    return VelocityMLP()


def test_euler_shape():
    model = _dummy_model()
    B, n_steps = 50, 40
    x0 = torch.randn(B, 2)
    traj = euler_integrate(model, x0, n_steps=n_steps)
    assert traj.shape == (n_steps + 1, B, 2), f"unexpected shape {traj.shape}"
    print("PASS  test_euler_shape")


def test_euler_initial_condition():
    model = _dummy_model()
    x0 = torch.randn(32, 2)
    traj = euler_integrate(model, x0, n_steps=20)
    assert torch.allclose(traj[0], x0), "trajectory[0] != x0"
    print("PASS  test_euler_initial_condition")


def test_rk4_shape():
    model = _dummy_model()
    B, n_steps = 50, 40
    x0 = torch.randn(B, 2)
    traj = rk4_integrate(model, x0, n_steps=n_steps)
    assert traj.shape == (n_steps + 1, B, 2)
    print("PASS  test_rk4_shape")


def test_constant_field_euler_vs_rk4():
    # For a constant velocity field v(x,t) = c, both solvers must reach x0 + c exactly.
    # Use a model whose output is always (1, 0) regardless of input.
    class ConstantModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.ones_like(x) * torch.tensor([1.0, 0.0])

    model = ConstantModel()
    x0 = torch.zeros(10, 2)
    n = 50

    euler_end = euler_integrate(model, x0, n_steps=n)[-1]
    rk4_end   = rk4_integrate(model, x0, n_steps=n)[-1]

    expected = torch.tensor([1.0, 0.0]).expand(10, 2)
    assert torch.allclose(euler_end, expected, atol=1e-5), f"euler endpoint wrong: {euler_end[0]}"
    assert torch.allclose(rk4_end,   expected, atol=1e-5), f"rk4 endpoint wrong: {rk4_end[0]}"
    print("PASS  test_constant_field_euler_vs_rk4")


def test_velocity_field_on_grid_shapes():
    model = _dummy_model()
    res = 15
    X, Y, U, V = velocity_field_on_grid(model, t=0.5, res=res)
    for name, arr in [("X", X), ("Y", Y), ("U", U), ("V", V)]:
        assert arr.shape == (res, res), f"{name} shape wrong: {arr.shape}"
    print("PASS  test_velocity_field_on_grid_shapes")


def test_velocity_field_finite():
    model = _dummy_model()
    X, Y, U, V = velocity_field_on_grid(model, t=0.3)
    assert torch.isfinite(U).all() and torch.isfinite(V).all()
    print("PASS  test_velocity_field_finite")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_euler_shape()
    test_euler_initial_condition()
    test_rk4_shape()
    test_constant_field_euler_vs_rk4()
    test_velocity_field_on_grid_shapes()
    test_velocity_field_finite()
    print("\nAll solver tests passed.")
