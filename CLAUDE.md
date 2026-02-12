# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EB-JEPA (Energy-Based Joint-Embedding Predictive Architectures) — a self-supervised representation learning library built on PyTorch 2.6.0. Implements JEPA models for images, video, and action-conditioned video planning. Also includes a Code JEPA module for code understanding via GNN on execution traces.

## Common Commands

```bash
# Install (uv recommended)
uv sync && source .venv/bin/activate

# Run examples (all use -m module syntax from repo root)
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml
python -m examples.video_jepa.main --fname examples/video_jepa/cfgs/default.yaml
python -m examples.ac_video_jepa.main --fname examples/ac_video_jepa/cfgs/train.yaml

# Override any YAML config param via CLI dot notation
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml optim.epochs=100 data.batch_size=128

# Tests
uv run pytest tests/
uv run pytest tests/test_jepa_output_formats.py -v    # single test file

# Linting (CI runs these on push to main)
python -m isort --check eb_jepa examples tests
python -m black --check eb_jepa examples tests

# Auto-format
autoflake --remove-all-unused-imports -r --in-place .
python -m isort eb_jepa examples tests
python -m black eb_jepa examples tests
```

## Architecture

### Core Library (`eb_jepa/`)

**Model hierarchy:** `JEPAbase` (inference) → `JEPA` (trainable) → `JEPAProbe` (frozen encoder + trainable head)

**Key method — `JEPA.unroll()`:** Central to training and planning. Supports two modes:
- **Parallel:** All timesteps at once (training). Pass `compute_loss=True`.
- **Autoregressive:** Step-by-step rollout (planning). Pass `compute_loss=False`.

**Components wired together in JEPA:**
- `encoder` — maps observations to latent space (ResNet5/18, ImpalaEncoder, ViT)
- `predictor` — predicts next latent state (ResUNet for conv, GRU-based RNNPredictor for sequential)
- `action_encoder` — encodes actions (Identity for simple spaces)
- `regularizer` — prevents representation collapse (VICReg, VCLoss, VC_IDM_Sim)
- `projector` — MLP projector; losses computed in projected space, not raw embedding space

**Key files:**
- `jepa.py` — model classes and unroll logic
- `architectures.py` — all encoder/predictor/projector networks
- `losses.py` — VICReg, BCS, VC, temporal, inverse dynamics losses
- `planning.py` — MPPI and CEM planners for action-conditioned JEPA

### Tensor Conventions

- Observations: `[B, C, T, H, W]`
- Actions: `[B, A, T]`
- Embeddings: `[B, D, T, 1, 1]` (singleton spatial dims for conv compatibility)

### Three Example Applications (`examples/`)

1. **image_jepa** — Self-supervised on CIFAR-10, evaluated via linear probing (~90% acc)
2. **video_jepa** — Future frame prediction on Moving MNIST
3. **ac_video_jepa** — World model + goal-conditioned planning in Two Rooms env (97% success)

Each example has `main.py` (training), `eval.py` (evaluation), and `cfgs/` (YAML configs).

### Code JEPA (`code_jepa/`)

GNN-based JEPA applied to code execution traces:
- `graph_tracer.py` — traces Python execution to build memory graphs
- `generate_graph_data.py` — converts traces to PyTorch Geometric graphs
- `train_gnn.py` — trains GNN encoder with JEPA objective on code graphs

### MVP (`mvp/`)

Earlier simplified Code JEPA implementation with training, evaluation, and anomaly detection.

## Configuration

YAML configs in `examples/*/cfgs/` control all hyperparameters. Environment variables:
- `EBJEPA_DSETS` — dataset directory
- `EBJEPA_CKPTS` — checkpoint directory

Default configs target H100 GPUs; reduce batch size for smaller GPUs.

## Training Pattern

All examples follow the same loop: `model.unroll()` returns `(preds, (total_loss, reg_loss, reg_unweight, reg_dict, pred_loss))`. AMP with bfloat16 and GradScaler used throughout. Wandb for experiment tracking.

## CI

GitHub Actions runs `isort --check` + `black --check` on pushes to main, and `pytest tests/` on all pushes.
