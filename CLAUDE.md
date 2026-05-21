# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Launch the Gradio UI (two tabs: toy distributions + MNIST latent flow)
python app.py                          # http://localhost:7860

# Pre-train the VAE before using the MNIST tab (saves checkpoints/vae.pt)
python scripts/train_vae.py            # ~2.5 min on CPU, 15 epochs

# Visualize the 2D MNIST latent space coloured by class (sanity check)
python scripts/eyeball_latent.py       # saves outputs/latent_space.png

# Per-module tests (run individually, no pytest required)
python tests/test_data.py
python tests/test_interpolation.py
python tests/test_model.py
python tests/test_train.py             # includes 5k-step CPU timing benchmark
python tests/test_solvers.py
python tests/test_vae.py               # 7 tests: shapes, loss, gradients

# Eyeball scripts (save PNGs to outputs/)
python scripts/eyeball_data.py
```

## Architecture

### Data flow

```
data.py → interpolation.py → model.py
                ↓                ↑
           train.py  ←── vae.py (MNIST latent sampler + class conditioning)
                ↓
          solvers.py
                ↓
         visualize.py
                ↓
            app.py  (Gradio, wires all modules together)
```

### Math conventions (used throughout)

- Source: `x0 ~ N(0, I)` — always sampled fresh each step, no dataset
- Target (toy): `x1 ~ TARGETS[name](batch_size)`
- Target (MNIST): `x1` sampled from pre-encoded latent pool for a given digit class
- Interpolant: `x_t = (1 - t) * x0 + t * x1`
- Ground-truth velocity: `x1 - x0` (constant along the path, independent of t)
- Loss: MSE between `model(x_t, t)` and `x1 - x0`

### Key conventions

**Trajectory shape**: `euler_integrate` and `rk4_integrate` both return `[n_steps+1, B, 2]` — index 0 is `x0`, index -1 is the final position. Visualize uses this shape directly for trail rendering.

**TARGETS registry** (`src/data.py`): a `dict[str, Callable[[int], Tensor]]` — add new toy distributions here and they appear automatically in the Gradio Tab 1 dropdown. MNIST latent samplers are built at runtime via `make_mnist_class_sampler(vae, digit)` and are not in this registry.

**In-memory model state** (`app.py`): `_state` dict holds `model`, `target` (toy tab) and `vae`, `mnist_model`, `mnist_class`, `mnist_sampler` (MNIST tab). All event handlers guard on `_state[key] is None`.

**Class conditioning** (`src/model.py`): `VelocityMLP(n_classes=10)` adds `nn.Embedding(10, 8)` and prepends the 8-dim embedding to the `[x, y, t]` input → 11-dim total. `n_classes=None` (default) is the original unconditioned model — all existing tests still pass.

**Layered rendering** (`src/visualize.py`): single axes, three artists stacked by `zorder`: quiver field (1, alpha=0.4) → trail `LineCollection` (2, alpha ramps 0→0.7 by age) → particle scatter (3, white). Trails are built by slicing `traj[frame-trail_len : frame+1]` each frame and constructing per-segment RGBA from `np.linspace`.

**Class-label broadcasting**: anywhere a `VelocityMLP` with `n_classes` is passed to `velocity_field_on_grid` or `euler_integrate` (which call `model(x, t)` with variable batch sizes), wrap it in a `_Bound` / `CWrapper` module that does `c[0:1].expand(x.shape[0])` — not `c[:x.shape[0]]`, which breaks when the grid has more points than `c`.

**Solver comparison**: `compare_solvers_figure` uses dense Euler (500 steps) as ground truth and reports mean per-particle L2 error for both Euler and RK4 at the same low step count. RK4 is ~34× more accurate at 15 steps on two_moons.

### VAE architecture (`src/vae.py`)

- **Encoder**: Conv2d(1→32→64, stride 2) → Flatten → Linear → `(mu [B,2], logvar [B,2])`
- **Decoder**: Linear(2→64×7×7) → Reshape → ConvTranspose2d(64→32→1) → Sigmoid
- **Loss**: BCE reconstruction + KL divergence (standard ELBO, summed over pixels, meaned over batch)
- **Checkpoint**: `checkpoints/vae.pt` (gitignored — must train or copy in manually)

### Training performance

- **Toy flow** — 5 000 steps, batch 256, CPU: ~15 seconds
- **VAE** — 15 epochs, batch 256, CPU: ~2.5 minutes (pre-loads all 60k images into a tensor, samples with `randint` — avoids DataLoader disk I/O)
- **MNIST latent flow** — 3 000 steps, batch 256, CPU: ~10 seconds

### Known platform notes

- **SSL cert error on Windows**: MNIST download fails with `CERTIFICATE_VERIFY_FAILED`. Fixed by patching `ssl._create_default_https_context = ssl._create_unverified_context` at import time in `src/vae.py` and `src/data.py`.
- **`torch.compile` unavailable**: requires MSVC (`cl.exe`) which is not on PATH. Would give ~2–3× speedup on the VAE if available.

## Current State

The MNIST latent flow tab is fully implemented and tested. Both tabs work end-to-end:

- **Tab 1** — toy distributions (two_moons, gaussian_mixture, checkerboard), quiver scrubber, GIF animation, Euler vs RK4 comparison
- **Tab 2** — VAE train/load, digit class selector (0–9), class-conditioned flow training, latent quiver, decoded digit panel, GIF animation

Demo GIFs are in `assets/` and embedded in the README.

## Possible Next Steps

- **More digit classes in one model**: train a single flow model conditioned on all 10 classes simultaneously rather than one model per digit
- **Better VAE**: increase latent dim beyond 2 (e.g. 8–16D) and use PCA/UMAP for visualization, or add a KL annealing schedule for cleaner clusters
- **Conditional interpolation**: let the user pick a source class and a target class and watch the field interpolate between them
- **Bigger dataset**: swap MNIST for FashionMNIST or CIFAR-10 (would need a larger VAE and higher-dim latent)
- **GPU support**: add `.to(device)` throughout and auto-detect CUDA/MPS
