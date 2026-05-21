"""Tests for the MNIST VAE (src/vae.py)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.vae import VAE, vae_loss, MNISTEncoder, MNISTDecoder


def test_encoder_output_shape():
    enc = MNISTEncoder()
    x = torch.randn(8, 1, 28, 28)
    mu, logvar = enc(x)
    assert mu.shape == (8, 2), f"Expected (8,2), got {mu.shape}"
    assert logvar.shape == (8, 2)
    print("PASS  encoder output shape")


def test_decoder_output_shape():
    dec = MNISTDecoder()
    z = torch.randn(8, 2)
    out = dec(z)
    assert out.shape == (8, 1, 28, 28), f"Expected (8,1,28,28), got {out.shape}"
    print("PASS  decoder output shape")


def test_decoder_output_range():
    dec = MNISTDecoder()
    z = torch.randn(32, 2)
    out = dec(z)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, "Decoder output outside [0,1]"
    print("PASS  decoder output in [0,1]")


def test_vae_forward_shapes():
    vae = VAE()
    x = torch.randn(16, 1, 28, 28).clamp(0, 1)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (16, 2)
    assert logvar.shape == (16, 2)
    print("PASS  VAE forward shapes")


def test_vae_loss_positive():
    vae = VAE()
    x = torch.rand(16, 1, 28, 28)
    recon, mu, logvar = vae(x)
    loss = vae_loss(recon, x, mu, logvar)
    assert loss.item() > 0, "VAE loss should be positive"
    assert torch.isfinite(loss), "VAE loss should be finite"
    print(f"PASS  vae_loss positive  (loss={loss.item():.2f})")


def test_vae_loss_decreases():
    torch.manual_seed(0)
    vae = VAE()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    x = torch.rand(64, 1, 28, 28)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "VAE loss should decrease with training"
    print(f"PASS  vae loss decreases ({losses[0]:.1f} -> {losses[-1]:.1f})")


def test_gradients_flow():
    vae = VAE()
    x = torch.rand(8, 1, 28, 28)
    recon, mu, logvar = vae(x)
    loss = vae_loss(recon, x, mu, logvar)
    loss.backward()
    for name, p in vae.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
    print("PASS  gradients flow to all parameters")


if __name__ == "__main__":
    test_encoder_output_shape()
    test_decoder_output_shape()
    test_decoder_output_range()
    test_vae_forward_shapes()
    test_vae_loss_positive()
    test_vae_loss_decreases()
    test_gradients_flow()
    print("\nAll VAE tests passed.")
