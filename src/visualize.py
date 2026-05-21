"""Visualization: layered quiver field + particle trails animation.

Rendering order (all on one axes, not subplots):
  zorder 1 — quiver field    (alpha=0.4, arrows coloured by magnitude)
  zorder 2 — trail segments  (LineCollection, alpha ramps oldest→0 newest→0.7)
  zorder 3 — particles       (white scatter, always on top)

The quiver field is re-evaluated at the current frame's time t so you can watch
the velocity field reorganize as t sweeps from 0 to 1.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import torch
from torch import Tensor

from src.model import VelocityMLP
from src.solvers import euler_integrate, velocity_field_on_grid


def render_static_quiver(
    model: VelocityMLP,
    t: float,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    res: int = 25,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Draw the learned velocity field at a fixed time t.

    Used by the Gradio timestep slider to show a static snapshot.

    Args:
        model:  Trained VelocityMLP.
        t:      Time in [0, 1] at which to evaluate the field.
        bounds: (x_min, x_max, y_min, y_max) of the plot region.
        res:    Quiver grid resolution.
        ax:     Existing axes to draw into; creates a new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#111111")
    else:
        fig = ax.figure

    X, Y, U, V = velocity_field_on_grid(model, t, bounds, res)
    X, Y = X.numpy(), Y.numpy()
    U, V = U.numpy(), V.numpy()
    mag = np.hypot(U, V)

    ax.set_facecolor("#111111")
    ax.quiver(X, Y, U, V, mag, cmap="plasma", alpha=0.4, zorder=1)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_title(f"velocity field   t = {t:.2f}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    return fig


def animate_flow(
    model: VelocityMLP,
    x0: Tensor,
    n_steps: int = 100,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    grid_res: int = 20,
    trail_len: int = 15,
    out_path: str = "outputs/flow.gif",
    fps: int = 20,
) -> str:
    """Animate particles flowing through the learned velocity field.

    Pre-computes the full trajectory and all quiver grids before rendering,
    so the FuncAnimation callback stays lightweight.

    Args:
        model:     Trained VelocityMLP.
        x0:        [B, 2] initial particle positions (noise samples).
        n_steps:   Number of Euler integration steps = number of frames.
        bounds:    Plot region (x_min, x_max, y_min, y_max).
        grid_res:  Quiver grid resolution per axis.
        trail_len: How many past frames each particle's trail spans.
        out_path:  Destination for the exported GIF.
        fps:       Playback speed of the saved GIF.

    Returns:
        Absolute path to the saved GIF.
    """
    model.eval()

    # --- Pre-compute trajectory -----------------------------------------------
    # traj[i, j] = position of particle j at step i.  Shape [n_steps+1, B, 2].
    traj = euler_integrate(model, x0, n_steps=n_steps).numpy()
    B = x0.shape[0]

    # --- Pre-compute quiver grids for every frame -----------------------------
    # Doing this up-front keeps the animation callback fast.
    quiver_frames: list[tuple[np.ndarray, ...]] = []
    with torch.no_grad():
        for i in range(n_steps + 1):
            t = i / n_steps
            X, Y, U, V = velocity_field_on_grid(model, t, bounds, grid_res)
            mag = np.hypot(U.numpy(), V.numpy())
            quiver_frames.append((X.numpy(), Y.numpy(), U.numpy(), V.numpy(), mag))

    # Global magnitude range for consistent coloring across all frames.
    all_mags = np.stack([f[4] for f in quiver_frames])
    vmin, vmax = float(all_mags.min()), float(all_mags.max())

    # --- Figure setup ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Quiver — layer 1 (background).  We'll call set_UVC each frame.
    X0, Y0, U0, V0, mag0 = quiver_frames[0]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    quiv = ax.quiver(
        X0, Y0, U0, V0, mag0,
        cmap="plasma", norm=norm,
        alpha=0.4, zorder=1,
        angles="xy", scale_units="xy", scale=1.5,
    )

    # Trail LineCollection — layer 2.
    lc = LineCollection([], linewidths=1.0, zorder=2)
    ax.add_collection(lc)

    # Particle scatter — layer 3 (foreground).
    scat = ax.scatter(
        traj[0, :, 0], traj[0, :, 1],
        s=10, c="white", zorder=3, linewidths=0,
    )

    title = ax.set_title("t = 0.00", color="white", pad=8)

    # --- Animation callback ---------------------------------------------------
    def _update(frame: int):
        # 1. Quiver: swap in pre-computed U/V for this frame's time.
        _, _, U, V, mag = quiver_frames[frame]
        quiv.set_UVC(U, V, mag)

        # 2. Trails: one fading segment chain per particle.
        segments: list[np.ndarray] = []
        rgba: list[tuple[float, float, float, float]] = []

        start = max(0, frame - trail_len)
        window = traj[start : frame + 1]          # [k, B, 2]
        k = window.shape[0]

        if k >= 2:
            # alpha ramps from 0 (oldest) to 0.7 (newest segment)
            alphas = np.linspace(0.0, 0.7, k - 1)
            for j in range(B):
                pts = window[:, j, :]             # [k, 2]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)  # [k-1, 2, 2]
                segments.extend(segs)
                rgba.extend((0.55, 0.78, 1.0, float(a)) for a in alphas)

        lc.set_segments(segments)
        lc.set_color(rgba)

        # 3. Particles: move to current positions.
        scat.set_offsets(traj[frame])

        t = frame / n_steps
        title.set_text(f"t = {t:.2f}")

        return quiv, lc, scat, title

    anim = FuncAnimation(
        fig, _update,
        frames=n_steps + 1,
        interval=1000 // fps,
        blit=False,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)

    return os.path.abspath(out_path)
