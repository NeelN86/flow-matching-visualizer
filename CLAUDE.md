# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Launch the Gradio UI
python app.py                          # http://localhost:7860

# Per-module tests (run individually, no pytest required)
python tests/test_data.py
python tests/test_interpolation.py
python tests/test_model.py
python tests/test_train.py             # includes 5k-step CPU timing benchmark
python tests/test_solvers.py

# Eyeball scripts (save PNGs to outputs/)
python scripts/eyeball_data.py
```

## Architecture

### Data flow

```
data.py → interpolation.py → model.py
                ↓
           train.py  ←─────────────────┐
                ↓                      │
          solvers.py          (retrain rebinds _state["model"])
                ↓
         visualize.py
                ↓
            app.py  (Gradio, wires all modules together)
```

### Math conventions (used throughout)

- Source: `x0 ~ N(0, I)` — always sampled fresh each step, no dataset
- Target: `x1 ~ TARGETS[name](batch_size)` — same
- Interpolant: `x_t = (1 - t) * x0 + t * x1`
- Ground-truth velocity: `x1 - x0` (constant along the path, independent of t)
- Loss: MSE between `model(x_t, t)` and `x1 - x0`

### Key conventions

**Trajectory shape**: `euler_integrate` and `rk4_integrate` both return `[n_steps+1, B, 2]` — index 0 is `x0`, index -1 is the final position. Visualize uses this shape directly for trail rendering.

**TARGETS registry** (`src/data.py`): a `dict[str, Callable[[int], Tensor]]` — add new distributions here and they appear automatically in the Gradio dropdown.

**In-memory model state** (`app.py`): `_state = {"model": None, "target": None}` — single global, updated on each Train click. All Gradio event handlers check `_state["model"] is None` before proceeding.

**Layered rendering** (`src/visualize.py`): single axes, three artists stacked by `zorder`: quiver field (1, alpha=0.4) → trail `LineCollection` (2, alpha ramps 0→0.7 by age) → particle scatter (3, white). Trails are built by slicing `traj[frame-trail_len : frame+1]` each frame and constructing per-segment RGBA from `np.linspace`.

**Solver comparison**: `compare_solvers_figure` uses dense Euler (500 steps) as ground truth and reports mean per-particle L2 error for both Euler and RK4 at the same low step count. RK4 is ~34× more accurate at 15 steps on two_moons.

### Training performance

5 000 steps, batch 256, CPU: ~15 seconds. No GPU needed. Distributions are sampled on-the-fly each step (infinite dataset, no DataLoader).

## Next Session Goal
Extend the toy visualizer into a class-conditioned image generator using 
a 2D VAE latent space trained on MNIST.

### Plan
1. Train a VAE with a 2D bottleneck on MNIST
2. Visualize the latent space — check digit classes cluster cleanly
3. Add class embedding to VelocityMLP input: [x, y, t, class_emb] -> [vx, vy]
4. Retrain flow matching in 2D latent space with class conditioning
5. Two-panel visualization: quiver field (left) + decoded image (right)
6. Class selector in Gradio — switching class reorganizes the velocity field live

### Key Design Decision
Work in 2D latent space not pixel space — keeps velocity field visualization 
exact rather than PCA-approximated.