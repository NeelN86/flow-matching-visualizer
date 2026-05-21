"""Eyeball test for src/data.py — render a 2x2 scatter grid of all distributions."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import torch

# Make `src` importable when running this file directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import (
    TARGETS,
    sample_source,
)

N = 2000
OUT_PATH = os.path.join(ROOT, "outputs", "data_eyeball.png")


def main() -> None:
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    panels = [("source N(0, I)", sample_source(N))]
    panels += [(name, fn(N)) for name, fn in TARGETS.items()]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (title, pts) in zip(axes.flat, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"data.py eyeball — {N} samples each", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
