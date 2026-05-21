"""Unit tests for src/interpolation.py."""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.interpolation import flow_matching_loss, interpolate, target_velocity


def test_interpolate_shapes():
    B = 64
    x0, x1 = torch.randn(B, 2), torch.randn(B, 2)
    t = torch.rand(B)
    x_t = interpolate(x0, x1, t)
    assert x_t.shape == (B, 2), f"expected ({B}, 2), got {x_t.shape}"
    print("PASS  test_interpolate_shapes")


def test_interpolate_boundaries():
    B = 32
    x0, x1 = torch.randn(B, 2), torch.randn(B, 2)

    # At t=0 the interpolant must equal x0 exactly.
    t0 = torch.zeros(B)
    assert torch.allclose(interpolate(x0, x1, t0), x0), "interpolate at t=0 != x0"

    # At t=1 the interpolant must equal x1 exactly.
    t1 = torch.ones(B)
    assert torch.allclose(interpolate(x0, x1, t1), x1), "interpolate at t=1 != x1"

    print("PASS  test_interpolate_boundaries")


def test_target_velocity():
    B = 16
    x0, x1 = torch.randn(B, 2), torch.randn(B, 2)
    v = target_velocity(x0, x1)
    assert v.shape == (B, 2)
    assert torch.allclose(v, x1 - x0)
    print("PASS  test_target_velocity")


def test_loss_zero_for_perfect_model():
    # A model that returns the exact target velocity should give loss = 0.
    B = 128
    x0, x1 = torch.randn(B, 2), torch.randn(B, 2)
    t = torch.rand(B)

    perfect_model = lambda x_t, t: target_velocity(x0, x1)
    loss = flow_matching_loss(perfect_model, x0, x1, t)
    assert loss.item() < 1e-6, f"expected ~0 loss, got {loss.item()}"
    print("PASS  test_loss_zero_for_perfect_model")


def test_loss_nonzero_for_bad_model():
    # A zero-predictor should have positive loss when x0 != x1.
    B = 128
    x0, x1 = torch.randn(B, 2), torch.randn(B, 2) + 5.0
    t = torch.rand(B)

    zero_model = lambda x_t, t: torch.zeros_like(x_t)
    loss = flow_matching_loss(zero_model, x0, x1, t)
    assert loss.item() > 0.1, f"expected positive loss, got {loss.item()}"
    print("PASS  test_loss_nonzero_for_bad_model")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_interpolate_shapes()
    test_interpolate_boundaries()
    test_target_velocity()
    test_loss_zero_for_perfect_model()
    test_loss_nonzero_for_bad_model()
    print("\nAll interpolation tests passed.")
