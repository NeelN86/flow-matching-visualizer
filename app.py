"""Gradio UI for the flow matching visualizer.

Single-page layout:
  Left column  — training controls + live loss curve
  Right column — velocity field (scrub t with slider)
  Bottom row   — animation controls + GIF output

A single in-memory model handle is updated each time the user clicks Train.
All heavy work (training, animation pre-computation) runs in the Gradio
request thread; for a single-user toy demo this is fine.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import gradio as gr

from src.data import TARGETS, sample_source
from src.interpolation import flow_matching_loss
from src.model import VelocityMLP
from src.visualize import animate_flow, render_static_quiver

# ---------------------------------------------------------------------------
# Shared mutable model state (single-user demo — no locking needed).
# ---------------------------------------------------------------------------
_state: dict = {"model": None, "target": None}


# ---------------------------------------------------------------------------
# Helper: loss figure
# ---------------------------------------------------------------------------

def _loss_fig(losses: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(losses, lw=0.6, alpha=0.45, color="#6ab0f5", label="per-step")
    if len(losses) >= 50:
        roll = np.convolve(losses, np.ones(50) / 50, mode="valid")
        ax.plot(range(49, len(losses)), roll, lw=2, color="#f5a623", label="50-step avg")
        ax.legend(fontsize=8, facecolor="#222", labelcolor="white", edgecolor="#444")
    ax.set_xlabel("step", color="white")
    ax.set_ylabel("MSE", color="white")
    ax.set_title("Training loss", color="white")
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def train_model(
    target_name: str,
    steps: int,
    batch_size: int,
    lr: float,
):
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
    """Re-render the quiver field at the chosen time t."""
    if _state["model"] is None:
        return None
    fig = render_static_quiver(_state["model"], t=float(t), res=25)
    return fig


def run_animate(n_particles: int, n_steps: int, trail_len: int) -> str | None:
    """Integrate particles and export a GIF, returned as a file path."""
    if _state["model"] is None:
        gr.Warning("Train the model first.")
        return None
    x0 = sample_source(int(n_particles))
    out = animate_flow(
        _state["model"],
        x0,
        n_steps=int(n_steps),
        trail_len=int(trail_len),
        out_path="outputs/flow.gif",
        fps=20,
    )
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Flow Matching Visualizer") as demo:
    gr.Markdown(
        "# Flow Matching Visualizer\n"
        "Train a velocity MLP `v_θ(x, t)` that transports Gaussian noise → a 2D target "
        "distribution, then watch the learned field and particle trajectories."
    )

    with gr.Row():
        # ── Left: training ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Training")
            target_dd = gr.Dropdown(
                choices=list(TARGETS.keys()),
                value="two_moons",
                label="Target distribution",
            )
            with gr.Row():
                steps_sl = gr.Slider(500, 10_000, value=5_000, step=500, label="Steps")
                batch_sl = gr.Slider(64, 512, value=256, step=64, label="Batch size")
            lr_sl = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="Learning rate")
            train_btn = gr.Button("Train", variant="primary")
            status_txt = gr.Textbox(label="Status", interactive=False, max_lines=1)
            loss_plot = gr.Plot(label="Live loss curve")

        # ── Right: velocity field ─────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Velocity field — scrub t to watch the field reorganize")
            t_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="t")
            quiver_plot = gr.Plot(label="Quiver field at t")

    # ── Bottom: animation ─────────────────────────────────────────────────
    gr.Markdown("### Animation")
    with gr.Row():
        np_sl = gr.Slider(50, 300, value=150, step=50, label="Particles")
        ns_sl = gr.Slider(20, 200, value=100, step=10, label="ODE steps (= frames)")
        tl_sl = gr.Slider(5, 30, value=15, step=5, label="Trail length (frames)")
    anim_btn = gr.Button("Animate + export GIF")
    gif_out = gr.Image(label="Flow animation", type="filepath")

    # ── Wiring ────────────────────────────────────────────────────────────
    train_btn.click(
        train_model,
        inputs=[target_dd, steps_sl, batch_sl, lr_sl],
        outputs=[loss_plot, status_txt],
    )
    t_slider.change(
        update_quiver,
        inputs=[t_slider],
        outputs=[quiver_plot],
    )
    anim_btn.click(
        run_animate,
        inputs=[np_sl, ns_sl, tl_sl],
        outputs=[gif_out],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
