"""Gradio UI for the flow matching visualizer.

Tab 1 — Toy distributions (original behaviour, untouched).
Tab 2 — MNIST latent flow:
  • Train a 2D-bottleneck VAE on MNIST (or load a saved checkpoint).
  • Choose a digit class (0–9), train class-conditioned flow matching
    in that 2D latent space.
  • Scrub the time slider to watch the velocity field reorganize.
  • Animate particles and decode their final positions to see digit images.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import gradio as gr

from src.data import TARGETS, sample_source, make_mnist_class_sampler
from src.interpolation import flow_matching_loss
from src.model import VelocityMLP
from src.train import train
from src.vae import VAE, train_vae, load_vae
from src.visualize import (
    animate_flow,
    compare_solvers_figure,
    render_decoded_panel,
    render_static_quiver,
)

# ---------------------------------------------------------------------------
# Shared mutable state (single-user demo — no locking needed).
# ---------------------------------------------------------------------------
_state: dict = {
    # Toy tab
    "model":  None,
    "target": None,
    # MNIST tab
    "vae":           None,
    "mnist_model":   None,
    "mnist_class":   None,
    "mnist_sampler": None,
}


# ---------------------------------------------------------------------------
# Helper: loss figure (shared between both tabs)
# ---------------------------------------------------------------------------

def _loss_fig(losses: list[float], title: str = "Training loss") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(losses, lw=0.6, alpha=0.45, color="#6ab0f5", label="per-step")
    if len(losses) >= 50:
        roll = np.convolve(losses, np.ones(50) / 50, mode="valid")
        ax.plot(range(49, len(losses)), roll, lw=2, color="#f5a623", label="50-step avg")
        ax.legend(fontsize=8, facecolor="#222", labelcolor="white", edgecolor="#444")
    ax.set_xlabel("step", color="white")
    ax.set_ylabel("loss", color="white")
    ax.set_title(title, color="white")
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Toy distributions
# ---------------------------------------------------------------------------

def train_model(target_name: str, steps: int, batch_size: int, lr: float):
    """Generator: trains the model and streams a live loss figure every 100 steps."""
    model = VelocityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    target_sampler = TARGETS[target_name]
    losses: list[float] = []

    model.train()
    for step in range(int(steps)):
        x0 = sample_source(int(batch_size))
        x1 = target_sampler(int(batch_size))
        t = torch.rand(int(batch_size))

        optimizer.zero_grad()
        loss = flow_matching_loss(model, x0, x1, t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == int(steps) - 1:
            fig = _loss_fig(losses)
            yield fig, f"Step {step + 1} / {int(steps)}"
            plt.close(fig)

    _state["model"] = model
    _state["target"] = target_name
    fig = _loss_fig(losses)
    yield fig, f"Done — trained on '{target_name}' for {int(steps)} steps."
    plt.close(fig)


def update_quiver(t: float) -> plt.Figure | None:
    if _state["model"] is None:
        return None
    fig = render_static_quiver(_state["model"], t=float(t), res=25)
    return fig


def run_animate(n_particles: int, n_steps: int, trail_len: int) -> str | None:
    if _state["model"] is None:
        gr.Warning("Train the model first.")
        return None
    x0 = sample_source(int(n_particles))
    return animate_flow(
        _state["model"], x0,
        n_steps=int(n_steps), trail_len=int(trail_len),
        out_path="outputs/flow.gif", fps=20,
    )


def run_solver_comparison(n_particles: int, n_steps: int) -> plt.Figure | None:
    if _state["model"] is None:
        gr.Warning("Train the model first.")
        return None
    x0 = sample_source(int(n_particles))
    return compare_solvers_figure(_state["model"], x0, n_steps=int(n_steps))


# ---------------------------------------------------------------------------
# Tab 2: MNIST latent flow
# ---------------------------------------------------------------------------

def train_vae_handler(epochs: int, batch_size: int, lr: float):
    """Generator: trains the VAE and streams per-epoch loss.

    Pre-loads all MNIST images into a single tensor, then samples minibatches
    with randint — avoids per-batch DataLoader disk I/O and is ~10x faster.
    """
    from src.vae import vae_loss, _load_mnist_tensor

    yield None, "Loading MNIST into memory..."
    all_imgs = _load_mnist_tensor("data")
    n = all_imgs.shape[0]
    bs = int(batch_size)
    steps_per_epoch = n // bs

    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    losses: list[float] = []

    model.train()
    for epoch in range(int(epochs)):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            idx  = torch.randint(0, n, (bs,))
            imgs = all_imgs[idx]
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / steps_per_epoch
        losses.append(avg)
        fig = _loss_fig(losses, title="VAE training loss (per epoch)")
        yield fig, f"Epoch {epoch + 1}/{int(epochs)}  loss={avg:.1f}"
        plt.close(fig)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vae.pt")
    _state["vae"] = model
    _state["vae"].eval()

    fig = _loss_fig(losses, title="VAE training loss (per epoch)")
    yield fig, f"Done — VAE trained for {int(epochs)} epochs. Saved to checkpoints/vae.pt"
    plt.close(fig)


def load_vae_handler() -> str:
    if not os.path.exists("checkpoints/vae.pt"):
        return "checkpoints/vae.pt not found — train the VAE first."
    _state["vae"] = load_vae("checkpoints/vae.pt")
    return "VAE loaded from checkpoints/vae.pt"


def train_mnist_flow(digit: int, steps: int, batch_size: int, lr: float):
    """Generator: trains class-conditioned flow matching in MNIST latent space."""
    if _state["vae"] is None:
        yield None, "Train or load the VAE first."
        return

    vae = _state["vae"]
    digit = int(digit)
    yield None, f"Building latent pool for digit {digit}..."

    sampler = make_mnist_class_sampler(vae, digit)
    _state["mnist_sampler"] = sampler

    model = VelocityMLP(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    c_tensor = torch.full((int(batch_size),), digit, dtype=torch.long)

    from src.interpolation import interpolate, target_velocity
    import torch.nn.functional as F

    losses: list[float] = []
    model.train()
    for step in range(int(steps)):
        x0 = sample_source(int(batch_size))
        x1 = sampler(int(batch_size))
        t  = torch.rand(int(batch_size))

        x_t      = interpolate(x0, x1, t)
        v_target = target_velocity(x0, x1)
        v_pred   = model(x_t, t, c_tensor)
        loss     = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == int(steps) - 1:
            fig = _loss_fig(losses, title=f"Flow loss — digit {digit}")
            yield fig, f"Step {step + 1}/{int(steps)}"
            plt.close(fig)

    _state["mnist_model"] = model
    _state["mnist_class"] = digit
    fig = _loss_fig(losses, title=f"Flow loss — digit {digit}")
    yield fig, f"Done — flow model trained for digit {digit}, {int(steps)} steps."
    plt.close(fig)


def update_mnist_quiver(t: float) -> plt.Figure | None:
    if _state["mnist_model"] is None:
        return None
    digit = _state["mnist_class"]
    c = torch.full((1,), digit, dtype=torch.long)

    class _Bound(torch.nn.Module):
        def __init__(self, m: VelocityMLP, c: torch.Tensor) -> None:
            super().__init__()
            self._m = m
            self._c = c
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self._m(x, t, self._c[0:1].expand(x.shape[0]))

    wrapped = _Bound(_state["mnist_model"], c)
    return render_static_quiver(wrapped, t=float(t), bounds=(-4, 4, -4, 4), res=20)


def run_mnist_decoded(n_particles: int, n_steps: int) -> plt.Figure | None:
    if _state["mnist_model"] is None or _state["vae"] is None:
        gr.Warning("Train both the VAE and the flow model first.")
        return None
    digit  = _state["mnist_class"]
    x0     = sample_source(int(n_particles))
    c_tens = torch.full((int(n_particles),), digit, dtype=torch.long)
    fig = render_decoded_panel(
        _state["mnist_model"], _state["vae"], x0,
        n_steps=int(n_steps),
        n_show=min(9, int(n_particles)),
        class_label=digit,
        c_tensor=c_tens,
    )
    return fig


def run_mnist_animate(n_particles: int, n_steps: int, trail_len: int) -> str | None:
    if _state["mnist_model"] is None:
        gr.Warning("Train the MNIST flow model first.")
        return None
    digit  = _state["mnist_class"]
    x0     = sample_source(int(n_particles))
    c_tens = torch.full((int(n_particles),), digit, dtype=torch.long)

    class _Bound(torch.nn.Module):
        def __init__(self, m: VelocityMLP, c: torch.Tensor) -> None:
            super().__init__()
            self._m = m
            self._c = c
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self._m(x, t, self._c[0:1].expand(x.shape[0]))

    wrapped = _Bound(_state["mnist_model"], c_tens)
    return animate_flow(
        wrapped, x0,
        n_steps=int(n_steps), trail_len=int(trail_len),
        bounds=(-4, 4, -4, 4),
        out_path="outputs/mnist_flow.gif", fps=20,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Flow Matching Visualizer") as demo:
    gr.Markdown(
        "# Flow Matching Visualizer\n"
        "**Tab 1** — toy 2D distributions.  "
        "**Tab 2** — class-conditioned generation in a 2D MNIST VAE latent space."
    )

    # =========================================================================
    # Tab 1: Toy distributions
    # =========================================================================
    with gr.Tab("Toy distributions"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Training")
                target_dd = gr.Dropdown(
                    choices=list(TARGETS.keys()), value="two_moons",
                    label="Target distribution",
                )
                with gr.Row():
                    steps_sl  = gr.Slider(500, 10_000, value=5_000, step=500,  label="Steps")
                    batch_sl  = gr.Slider(64,  512,    value=256,   step=64,   label="Batch size")
                lr_sl     = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="Learning rate")
                train_btn = gr.Button("Train", variant="primary")
                status_txt = gr.Textbox(label="Status", interactive=False, max_lines=1)
                loss_plot  = gr.Plot(label="Live loss curve")

            with gr.Column(scale=1):
                gr.Markdown("### Velocity field — scrub t")
                t_slider    = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="t")
                quiver_plot = gr.Plot(label="Quiver field at t")

        gr.Markdown("### Animation")
        with gr.Row():
            np_sl  = gr.Slider(50,  300, value=150, step=50, label="Particles")
            ns_sl  = gr.Slider(20,  200, value=100, step=10, label="ODE steps")
            tl_sl  = gr.Slider(5,   30,  value=15,  step=5,  label="Trail length")
        anim_btn = gr.Button("Animate + export GIF")
        gif_out  = gr.Image(label="Flow animation", type="filepath")

        gr.Markdown("### Euler vs RK4 — solver comparison")
        with gr.Row():
            cmp_p_sl = gr.Slider(100, 500, value=300, step=100, label="Particles")
            cmp_s_sl = gr.Slider(5,   50,  value=15,  step=5,   label="ODE steps (keep low)")
        cmp_btn  = gr.Button("Compare solvers")
        cmp_plot = gr.Plot(label="Euler | RK4 | Ground truth")

        train_btn.click(train_model, [target_dd, steps_sl, batch_sl, lr_sl],
                        [loss_plot, status_txt])
        t_slider.change(update_quiver, [t_slider], [quiver_plot])
        anim_btn.click(run_animate, [np_sl, ns_sl, tl_sl], [gif_out])
        cmp_btn.click(run_solver_comparison, [cmp_p_sl, cmp_s_sl], [cmp_plot])

    # =========================================================================
    # Tab 2: MNIST latent flow
    # =========================================================================
    with gr.Tab("MNIST latent flow"):
        gr.Markdown(
            "### Step 1 — Train or load a 2D-bottleneck VAE on MNIST\n"
            "The VAE encodes each 28×28 digit into a 2D point. "
            "Flow matching then learns to transport Gaussian noise to the class cluster."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    vae_epochs_sl = gr.Slider(5,  30, value=15, step=5,   label="Epochs")
                    vae_batch_sl  = gr.Slider(64, 512, value=256, step=64, label="Batch size")
                vae_lr_sl      = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="LR")
                with gr.Row():
                    vae_train_btn = gr.Button("Train VAE", variant="primary")
                    vae_load_btn  = gr.Button("Load saved VAE")
                vae_status  = gr.Textbox(label="VAE status", interactive=False, max_lines=1)
                vae_loss_pl = gr.Plot(label="VAE loss curve")

        vae_train_btn.click(
            train_vae_handler,
            [vae_epochs_sl, vae_batch_sl, vae_lr_sl],
            [vae_loss_pl, vae_status],
        )
        vae_load_btn.click(load_vae_handler, [], [vae_status])

        gr.Markdown("### Step 2 — Train class-conditioned flow matching")
        with gr.Row():
            digit_dd      = gr.Dropdown(choices=list(range(10)), value=3, label="Digit class")
            flow_steps_sl = gr.Slider(500, 10_000, value=3_000, step=500,  label="Steps")
            flow_batch_sl = gr.Slider(64,  512,    value=256,   step=64,   label="Batch size")
            flow_lr_sl    = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4,  label="LR")
        flow_train_btn = gr.Button("Train flow model", variant="primary")
        flow_status    = gr.Textbox(label="Flow status", interactive=False, max_lines=1)
        flow_loss_pl   = gr.Plot(label="Flow loss curve")

        flow_train_btn.click(
            train_mnist_flow,
            [digit_dd, flow_steps_sl, flow_batch_sl, flow_lr_sl],
            [flow_loss_pl, flow_status],
        )

        gr.Markdown("### Step 3 — Visualize")
        with gr.Row():
            mt_slider   = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="t (velocity field)")
            m_np_sl     = gr.Slider(9,   100, value=25,  step=8,    label="Particles")
            m_ns_sl     = gr.Slider(20,  200, value=100, step=10,   label="ODE steps")
        with gr.Row():
            m_quiver_btn  = gr.Button("Update quiver")
            m_decoded_btn = gr.Button("Show decoded digits")
        m_quiver_pl   = gr.Plot(label="Latent velocity field at t")
        m_decoded_pl  = gr.Plot(label="Quiver + decoded digits")

        gr.Markdown("#### Animation")
        with gr.Row():
            m_tl_sl   = gr.Slider(5, 30, value=15, step=5, label="Trail length")
        m_anim_btn  = gr.Button("Animate + export GIF")
        m_gif_out   = gr.Image(label="MNIST flow animation", type="filepath")

        mt_slider.change(update_mnist_quiver, [mt_slider], [m_quiver_pl])
        m_quiver_btn.click(update_mnist_quiver, [mt_slider], [m_quiver_pl])
        m_decoded_btn.click(run_mnist_decoded, [m_np_sl, m_ns_sl], [m_decoded_pl])
        m_anim_btn.click(run_mnist_animate, [m_np_sl, m_ns_sl, m_tl_sl], [m_gif_out])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
