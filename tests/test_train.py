"""Tests for src/train.py."""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from src.model import VelocityMLP
from src.train import train


def test_return_types():
    model, history = train(target_name="two_moons", steps=10, batch_size=64)
    assert isinstance(model, VelocityMLP)
    assert isinstance(history, list) and len(history) == 10
    assert all(isinstance(v, float) for v in history)
    print("PASS  test_return_types")


def test_loss_decreases():
    # Over 300 steps the loss should drop — averaged early vs late to smooth noise.
    torch.manual_seed(0)
    _, history = train(target_name="two_moons", steps=300, batch_size=256)
    early = sum(history[:30]) / 30
    late = sum(history[-30:]) / 30
    assert late < early, f"loss did not decrease: early={early:.4f}, late={late:.4f}"
    print(f"PASS  test_loss_decreases  (early={early:.4f} -> late={late:.4f})")


def test_all_targets_run():
    for name in ("two_moons", "gaussian_mixture", "checkerboard"):
        _, history = train(target_name=name, steps=5, batch_size=32)
        assert len(history) == 5
    print("PASS  test_all_targets_run")


def test_progress_callback():
    calls: list[tuple[int, float]] = []
    train(steps=20, batch_size=32, progress_cb=lambda s, l: calls.append((s, l)))
    assert len(calls) == 20
    assert calls[0][0] == 0 and calls[-1][0] == 19
    print("PASS  test_progress_callback")


def test_cpu_timing():
    # Full 5k-step run must complete under 60 seconds on CPU.
    torch.manual_seed(1)
    t0 = time.perf_counter()
    _, history = train(steps=5_000, batch_size=256)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"training took {elapsed:.1f}s — exceeds 60s budget"
    print(f"PASS  test_cpu_timing  ({elapsed:.1f}s for 5000 steps)")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_return_types()
    test_loss_decreases()
    test_all_targets_run()
    test_progress_callback()
    test_cpu_timing()
    print("\nAll training tests passed.")
