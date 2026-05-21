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
from src.solvers import euler_integrate, rk4_integrate, velocity_field_on_grid

# VAE import is optional — only needed for render_decoded_panel.
try:
    from src.vae import VAE as _VAE
except ImportError:
    _VAE = None  # type: ignore[assignment,misc]


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


def render_decoded_panel(
    model: VelocityMLP,
    vae: "_VAE",
    x0: Tensor,
    n_steps: int = 100,
    n_show: int = 9,
    class_label: int | None = None,
    bounds: tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
    res: int = 20,
    c_tensor: Tensor | None = None,
) -> plt.Figure:
    """Two-panel figure: quiver field (left) + decoded MNIST images (right).

    Integrates n_show particles from x0, decodes their final latent positions
    with the VAE decoder, and shows a grid of reconstructed 28×28 images.

    Args:
        model:       Trained class-conditioned VelocityMLP.
        vae:         Trained VAE for decoding latent → pixel.
        x0:          [B, 2] initial noise positions.
        n_steps:     ODE integration steps.
        n_show:      How many decoded images to show (≤ B, arranged in a grid).
        class_label: Digit class, shown in the title.
        bounds:      Quiver field plot region.
        res:         Quiver grid resolution.
        c_tensor:    [B] class label tensor for the flow model.

    Returns:
        matplotlib Figure with two panels.
    """
    model.eval()
    vae.eval()

    # Integrate — use a wrapper that binds c_tensor if provided.
    if c_tensor is not None:
        class CWrapper(torch.nn.Module):
            def __init__(self, m: VelocityMLP, c: Tensor) -> None:
                super().__init__()
                self._m = m
                self._c = c
            def forward(self, x: Tensor, t: Tensor) -> Tensor:
                c = self._c[0:1].expand(x.shape[0])
                return self._m(x, t, c)
        wrapped = CWrapper(model, c_tensor)
    else:
        wrapped = model  # type: ignore[assignment]

    with torch.no_grad():
        traj = euler_integrate(wrapped, x0, n_steps=n_steps)  # [n_steps+1, B, 2]
        z_final = traj[-1, :n_show]                           # [n_show, 2]
        decoded = vae.decode(z_final)                         # [n_show, 1, 28, 28]

    # --- Figure layout --------------------------------------------------------
    grid_side = int(np.ceil(np.sqrt(n_show)))
    fig = plt.figure(figsize=(13, 6))
    fig.patch.set_facecolor("#111111")

    # Left: quiver field at t=1.0 (the learned target region).
    ax_q = fig.add_subplot(1, 2, 1)
    X, Y, U, V = velocity_field_on_grid(model if c_tensor is None else wrapped, 1.0, bounds, res)
    X, Y = X.numpy(), Y.numpy()
    U, V = U.numpy(), V.numpy()
    mag = np.hypot(U, V)
    ax_q.set_facecolor("#111111")
    ax_q.quiver(X, Y, U, V, mag, cmap="plasma", alpha=0.4, zorder=1)
    z_np = z_final.numpy()
    ax_q.scatter(z_np[:, 0], z_np[:, 1], s=20, c="white", zorder=2)
    ax_q.set_xlim(bounds[0], bounds[1])
    ax_q.set_ylim(bounds[2], bounds[3])
    ax_q.set_aspect("equal")
    title_str = f"Latent field  (class {class_label})" if class_label is not None else "Latent field"
    ax_q.set_title(title_str, color="white")
    ax_q.tick_params(colors="white")
    for sp in ax_q.spines.values():
        sp.set_edgecolor("#444444")

    # Right: grid of decoded images.
    ax_d = fig.add_subplot(1, 2, 2)
    ax_d.set_facecolor("#111111")
    ax_d.axis("off")
    imgs_np = decoded.squeeze(1).numpy()   # [n_show, 28, 28]
    canvas = np.zeros((grid_side * 28, grid_side * 28))
    for idx in range(n_show):
        r, c = divmod(idx, grid_side)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = imgs_np[idx]
    ax_d.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax_d.set_title("Decoded digits", color="white")

    fig.tight_layout()
    return fig


def compare_solvers_figure(
    model: VelocityMLP,
    x0: Tensor,
    n_steps: int = 20,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    gt_steps: int = 500,
) -> plt.Figure:
    """Side-by-side comparison of Euler vs RK4 at the same (low) step count.

    A high-step Euler run (gt_steps) serves as the ground truth. Each panel
    shows the final particle positions and the mean endpoint L2 error vs that
    reference, making the higher-order accuracy of RK4 tangible at few steps.

    Args:
        model:    Trained VelocityMLP.
        x0:       [B, 2] shared initial noise — both solvers start here.
        n_steps:  Steps for the cheap Euler and RK4 runs (keep low, e.g. 5–30).
        bounds:   Plot region.
        gt_steps: Steps for the reference trajectory (should be >> n_steps).

    Returns:
        A 1×3 matplotlib Figure: Euler | RK4 | Ground truth (gt_steps Euler).
    """
    # Ground truth: dense Euler integration treated as reference.
    gt = euler_integrate(model, x0, n_steps=gt_steps)[-1].numpy()   # [B, 2]
    euler_end = euler_integrate(model, x0, n_steps=n_steps)[-1].numpy()
    rk4_end   = rk4_integrate(model, x0, n_steps=n_steps)[-1].numpy()

    # Mean per-particle L2 distance from ground truth.
    err_euler = float(np.linalg.norm(euler_end - gt, axis=1).mean())
    err_rk4   = float(np.linalg.norm(rk4_end   - gt, axis=1).mean())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#111111")

    panels = [
        (euler_end, f"Euler  ({n_steps} steps)\nmean err = {err_euler:.4f}", "#6ab0f5"),
        (rk4_end,   f"RK4    ({n_steps} steps)\nmean err = {err_rk4:.4f}",   "#f5a623"),
        (gt,        f"Reference\n(Euler {gt_steps} steps)",                    "#aaffaa"),
    ]

    x_min, x_max, y_min, y_max = bounds
    for ax, (pts, title, color) in zip(axes, panels):
        ax.set_facecolor("#111111")
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c=color, alpha=0.7)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    fig.suptitle(
        f"Solver comparison — {n_steps} steps  |  RK4 err / Euler err = {err_rk4/max(err_euler,1e-9):.3f}",
        color="white", fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig
