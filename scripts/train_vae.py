"""Standalone script: pre-train the MNIST VAE and save checkpoints/vae.pt.

Run from repo root:
    python scripts/train_vae.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.vae import train_vae


def main() -> None:
    print("Training MNIST VAE (2D latent) ...")

    def cb(epoch: int, loss: float) -> None:
        print(f"  epoch {epoch + 1:>3}  loss = {loss:.2f}")

    _, history = train_vae(epochs=15, batch_size=256, lr=1e-3, progress_cb=cb)
    print(f"\nFinal loss: {history[-1]:.2f}")
    print("Saved → checkpoints/vae.pt")


if __name__ == "__main__":
    main()
