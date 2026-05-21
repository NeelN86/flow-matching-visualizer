"""VAE with a 2D bottleneck trained on MNIST.

Encoder: Conv2d(1→32→64) → Flatten → Linear → (mu [B,2], logvar [B,2])
Decoder: Linear(2 → 64*7*7) → Reshape → ConvTranspose2d(64→32→1) → Sigmoid

The 2D latent space lets us visualize and run flow matching directly on
latent coordinates without PCA approximation.
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # → [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # → [B, 64,  7,  7]
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(64 * 7 * 7, 2)
        self.fc_logvar = nn.Linear(64 * 7 * 7, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class MNISTDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # → [B, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32,  1, 4, stride=2, padding=1),  # → [B,  1, 28, 28]
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = MNISTEncoder()
        self.decoder = MNISTDecoder()

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / x.shape[0]
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    return bce + kld


def train_vae(
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    data_root: str = "data",
    progress_cb: Callable[[int, float], None] | None = None,
) -> tuple[VAE, list[float]]:
    """Train the VAE on MNIST and return (model, per-epoch loss history).

    Saves checkpoint to checkpoints/vae.pt after training.
    """
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        loss_history.append(avg)

        if progress_cb is not None:
            progress_cb(epoch, avg)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vae.pt")

    return model, loss_history


def load_vae(path: str = "checkpoints/vae.pt") -> VAE:
    model = VAE()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
