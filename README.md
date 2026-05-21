# Flow Matching Visualizer

A minimal, interactive toy that trains a small MLP to learn a **velocity field** that transports 2D Gaussian noise toward a target distribution — then lets you watch it work in real time.

Built as a focused learning project to understand the mechanics of [flow matching](https://arxiv.org/abs/2210.02747) before working on generative model infrastructure.

---

## What is flow matching?

Flow matching is a simulation-free method for training continuous normalizing flows. Instead of simulating an ODE during training, it directly regresses a neural network onto a known vector field that transports samples from a source distribution to a target.

### The math in three lines

Given a source sample **x₀ ~ N(0, I)** and a target sample **x₁ ~ p_data**, define a straight-line path:

```
x(t) = (1 − t)·x₀  +  t·x₁       t ∈ [0, 1]
```

The velocity along this path is constant:

```
dx/dt = x₁ − x₀
```

We train a neural network **v_θ(x, t)** to match this velocity at arbitrary (x, t) pairs:

```
L(θ) = E[ ‖ v_θ(x_t, t) − (x₁ − x₀) ‖² ]
```

At inference time, integrating **dx/dt = v_θ(x, t)** from t = 0 to t = 1 transports a noise sample to the learned target distribution.

---

## Project structure

```
src/
  data.py          — source (Gaussian) + 3 target distribution samplers
  interpolation.py — linear interpolant, target velocity, MSE loss
  model.py         — VelocityMLP: 3×256 hidden layers, SiLU activation
  train.py         — Adam training loop (~5k steps, runs on CPU in ~15s)
  solvers.py       — Euler and RK4 ODE integrators, grid velocity helper
  visualize.py     — layered quiver + particle trail animation, GIF export
app.py             — Gradio UI: distribution picker, timestep slider,
                     live loss curve, animate button, GIF export
tests/             — per-module unit tests (shape, finite, gradient flow)
```

---

## Supported target distributions

| Key | Description |
|---|---|
| `two_moons` | Two interlocking crescents |
| `gaussian_mixture` | 8 Gaussian modes on a circle |
| `checkerboard` | 4×4 alternating squares in [−2, 2]² |

---

## Quick start

```bash
pip install -r requirements.txt
python app.py          # launches Gradio UI at http://localhost:7860
```

To train and generate samples from the command line:

```bash
python -c "
from src.train import train
from src.solvers import euler_integrate
from src.data import sample_source
import torch

torch.manual_seed(0)
model, loss = train('two_moons', steps=5000)
x0 = sample_source(500)
traj = euler_integrate(model, x0)
print('final positions shape:', traj[-1].shape)
"
```

To run all tests:

```bash
python tests/test_data.py
python tests/test_interpolation.py
python tests/test_model.py
python tests/test_train.py
python tests/test_solvers.py
```

---

## Stack

- **PyTorch** — model, training, tensor ops (CPU-only, no GPU required)
- **torchdiffeq** — reference ODE solver for comparison
- **Matplotlib** — `FuncAnimation` for the layered quiver + particle visualization
- **Gradio** — interactive UI
- **imageio** — GIF export
