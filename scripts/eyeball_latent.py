"""Sanity-check: encode MNIST test set and scatter-plot the 2D latent space.

Saves outputs/latent_space.png — 10 digit classes should form distinct clusters.

Run from repo root:
    python scripts/eyeball_latent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.vae import load_vae


COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]


def main() -> None:
    if not os.path.exists("checkpoints/vae.pt"):
        print("checkpoints/vae.pt not found — run scripts/train_vae.py first.")
        sys.exit(1)

    vae = load_vae()

    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)

    all_mu:    list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, labels in loader:
            mu, _ = vae.encode(imgs)
            all_mu.append(mu)
            all_labels.append(labels)

    mu     = torch.cat(all_mu).numpy()
    labels = torch.cat(all_labels).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    for digit in range(10):
        mask = labels == digit
        ax.scatter(mu[mask, 0], mu[mask, 1], s=4, alpha=0.5,
                   color=COLORS[digit], label=str(digit))

    ax.legend(
        title="Digit", title_fontsize=9, fontsize=8,
        facecolor="#222", labelcolor="white", edgecolor="#444",
        markerscale=3,
    )
    ax.set_title("MNIST 2D latent space (VAE μ)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/latent_space.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
