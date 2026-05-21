"""Source and target distribution samplers for the flow-matching toy.

The flow matches a source distribution (standard Gaussian) to a target
distribution chosen from the TARGETS registry. Each sampler returns a
float32 tensor of shape [n, 2].
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor


def sample_source(n: int) -> Tensor:
    # x0 ~ N(0, I) in R^2 — the noise we transport toward the target.
    return torch.randn(n, 2)


def sample_two_moons(n: int, noise: float = 0.05) -> Tensor:
    n0 = n // 2
    n1 = n - n0

    # Upper moon: half-circle of radius 1 centered at origin, theta in [0, pi].
    theta0 = torch.rand(n0) * math.pi
    moon0 = torch.stack([torch.cos(theta0), torch.sin(theta0)], dim=1)

    # Lower moon: mirrored and shifted so the two crescents interlock.
    theta1 = torch.rand(n1) * math.pi
    moon1 = torch.stack([1.0 - torch.cos(theta1), 0.5 - torch.sin(theta1)], dim=1)

    pts = torch.cat([moon0, moon1], dim=0)
    pts = pts + noise * torch.randn_like(pts)

    # Recenter so the distribution is roughly symmetric about the origin.
    pts = pts - pts.mean(dim=0, keepdim=True)
    return pts


def sample_gaussian_mixture(
    n: int, k: int = 8, radius: float = 4.0, std: float = 0.3
) -> Tensor:
    # k modes evenly spaced on a circle of given radius.
    angles = torch.linspace(0.0, 2.0 * math.pi, k + 1)[:-1]
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # [k, 2]

    # Uniformly assign each sample to one mode, then add isotropic noise.
    assignments = torch.randint(0, k, (n,))
    return centers[assignments] + std * torch.randn(n, 2)


def sample_checkerboard(n: int) -> Tensor:
    # 4x4 checker over [-2, 2]^2 — keep only squares where floor(x)+floor(y) is even.
    # Rejection sampling: oversample uniformly, then mask. Loop until we have n points.
    collected: list[Tensor] = []
    have = 0
    while have < n:
        batch = torch.rand(2 * n, 2) * 4.0 - 2.0  # uniform in [-2, 2]^2
        keep_mask = ((torch.floor(batch[:, 0]) + torch.floor(batch[:, 1])) % 2 == 0)
        kept = batch[keep_mask]
        collected.append(kept)
        have += kept.shape[0]
    return torch.cat(collected, dim=0)[:n]


# Registry consumed by the Gradio dropdown and the training loop.
# Each entry is a callable f(n) -> Tensor[n, 2].
TARGETS: dict[str, Callable[[int], Tensor]] = {
    "two_moons": sample_two_moons,
    "gaussian_mixture": sample_gaussian_mixture,
    "checkerboard": sample_checkerboard,
}


def make_mnist_class_sampler(vae: "VAE", digit: int, data_root: str = "data") -> Callable[[int], Tensor]:  # type: ignore[name-defined]
    """Pre-encode all MNIST training images for a given digit into 2D latents.

    Returns a sampler f(n) -> Tensor[n, 2] that draws random rows from
    that pre-encoded pool — used as the flow-matching target distribution.
    The VAE is required at call time (not import time), so this never
    pollutes the TARGETS registry with a model dependency.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)

    latents: list[Tensor] = []
    vae.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            mask = labels == digit
            if mask.any():
                mu, _ = vae.encode(imgs[mask])
                latents.append(mu)

    pool = torch.cat(latents)  # [N_digit, 2]

    def sampler(n: int) -> Tensor:
        idx = torch.randint(0, pool.shape[0], (n,))
        return pool[idx]

    return sampler
