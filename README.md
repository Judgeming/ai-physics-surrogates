# AI Physics Surrogates

Neural-network surrogates for physics simulations, built step by step: PyTorch fundamentals,
neural operators and physics-informed networks, an end-to-end simulation-to-surrogate pipeline
for chip thermal analysis, and an LLM agent that drives simulations with verifiable results.

Independent personal project. Public data and open-source tools only.

## Roadmap

| Stage | Content | Status |
|---|---|---|
| P0 | [PyTorch training fundamentals](p0_pytorch_basics/README.md) on an analytical heat-conduction problem | in progress |
| P1 | Fourier Neural Operator on 2D Darcy flow; PINN for 2D heat conduction | planned |
| P2 | Chip thermal surrogate: FEM simulation campaign, neural-operator training, surrogate-in-the-loop Bayesian optimization | planned |
| P3 | Agentic simulation assistant with retrieval and a solver-verified evaluation harness | planned |

## Setup

Requires [uv](https://docs.astral.sh/uv/). PyTorch is pulled from the CUDA 12.8 index
(needed for RTX 50-series GPUs; CPU also works).

```bash
uv sync
uv run pytest
```
