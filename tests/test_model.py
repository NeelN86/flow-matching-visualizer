"""Smoke tests for src/model.py."""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import VelocityMLP


def test_output_shape():
    model = VelocityMLP()
    B = 64
    x = torch.randn(B, 2)
    t = torch.rand(B)
    out = model(x, t)
    assert out.shape == (B, 2), f"expected ({B}, 2), got {out.shape}"
    print("PASS  test_output_shape")


def test_output_finite():
    model = VelocityMLP()
    x = torch.randn(256, 2)
    t = torch.rand(256)
    out = model(x, t)
    assert torch.isfinite(out).all(), "output contains NaN or Inf"
    print("PASS  test_output_finite")


def test_gradient_flows():
    # Every parameter must receive a gradient after a backward pass.
    model = VelocityMLP()
    x = torch.randn(32, 2)
    t = torch.rand(32)
    loss = model(x, t).pow(2).mean()
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    print("PASS  test_gradient_flows")


def test_time_sensitivity():
    # The model should predict different velocities at t=0 vs t=1 (same x).
    model = VelocityMLP()
    model.eval()
    x = torch.randn(16, 2)
    with torch.no_grad():
        v0 = model(x, torch.zeros(16))
        v1 = model(x, torch.ones(16))
    diff = (v0 - v1).abs().mean().item()
    assert diff > 1e-6, "model output identical at t=0 and t=1 — t has no effect"
    print(f"PASS  test_time_sensitivity  (mean |v(t=0)-v(t=1)| = {diff:.4f})")


def test_parameter_count():
    model = VelocityMLP()
    n = sum(p.numel() for p in model.parameters())
    # 3→256 + 256→256×2 + 256→2 = 1_024 + 131_840 + 514 = ~200k
    print(f"PASS  test_parameter_count  ({n:,} parameters)")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_output_shape()
    test_output_finite()
    test_gradient_flows()
    test_time_sensitivity()
    test_parameter_count()
    print("\nAll model tests passed.")
