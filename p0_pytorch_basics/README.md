# P0 - PyTorch training fundamentals

**Goal:** write a complete, reproducible PyTorch training loop from a blank file without looking
anything up. **Budget:** 6-10 hours.

Every exercise uses one physics problem with a known answer: a 1D slab with uniform heat
generation `q`, conductivity `k`, and convective cooling `h` on both faces
(see [`physics.py`](physics.py)). Because the analytical solution is exact, every error you
measure comes from your model or your code, never from the data.

## How to work

```bash
cd E:\ai-physics-surrogates
uv run pytest -k ex01                                  # check one exercise
uv run python -m p0_pytorch_basics.ex01_autograd_fit   # run it
```

1. Read the exercise file's docstrings: they are the instructions.
2. Replace each `raise NotImplementedError` with your code.
3. Iterate until `uv run pytest -k exNN` is green, then run the module and read its output.
4. Only then compare with `solutions/`. To see the reference pass: `P0_SOLUTIONS=1 uv run pytest`
   (PowerShell: `$env:P0_SOLUTIONS=1; uv run pytest; Remove-Item Env:P0_SOLUTIONS`).

## Exercises

| # | File | Time | You practice |
|---|------|------|--------------|
| 1 | `ex01_autograd_fit.py` | 1-1.5 h | Autograd, manual gradient descent, `torch.optim`. Solve an inverse problem: recover `q`, `h` from noisy thermocouples |
| 2 | `ex02_mlp_module.py` | 0.5-1 h | `nn.Module`, `nn.Sequential`, parameter registration and counting |
| 3 | `ex03_dataset.py` | 1-1.5 h | `Dataset`/`DataLoader`, log-space targets, normalization without train/val leakage |
| 4 | `ex04_train.py` | 2-3 h | Full training script: seeding, devices, train/eval loops, LR scheduling, early stopping, checkpoints, TensorBoard, engineering metrics in deg C |
| 5 | `ex05_ood_dimensionless.py` | 1.5-2 h | Compact loop from memory, out-of-distribution testing, dimensional analysis as an inductive bias |
| - | Blank-file challenge (below) | 0.5 h | Proof that P0 is done |

## Companion lessons

Concept background from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch).
Read the concept sections and run the lesson code; the exercises here are where you practice.

| When | Lesson | Why |
|---|---|---|
| Before ex 1 | [Phase 3.06 Optimizers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/06-optimizers) | Explains why plain gradient descent diverges at a large step and what Adam changes |
| Before ex 2 | [Phase 3.11 Introduction to PyTorch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/11-intro-to-pytorch) | Tensors, `nn.Module`, training loop, `DataLoader`, GPU in one pass (on MNIST classification) |
| During ex 4 | [Phase 3.13 Debugging Neural Networks](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/13-debugging-neural-networks) | Overfit-one-batch test, learning-rate finder, gradient checks. Apply them to your ex 4 model |
| Optional | Phase 3.09 LR Schedules, Phase 1.13 Numerical Stability, Phase 2.13 ML Pipelines (leakage + reproducibility sections), Phase 1.05 Chain Rule & Autodiff | Reference when a topic in ex 3-4 feels shaky |

## Things to notice

- **Ex 1:** with `lr=4e-3` plain gradient descent no longer converges; with `2e-3` it does. Work
  out why from the loss curvature. Both optimizers also end ~2% from the true values. That gap is
  the noise-limited least-squares optimum, not an optimizer failure. How would you put an
  uncertainty on `q` and `h`?
- **Ex 4:** time the run on `--device cpu` and on `--device cuda`. For a ~9k-parameter MLP the CPU
  is usually as fast or faster (GPU kernel-launch overhead dominates tiny batches). The GPU pays
  off in P1/P2.
- **Ex 4:** the plotted profiles are not exactly symmetric in `x/L`, even though the physics is.
  The network does not know that. Try feeding `xi**2` instead of `xi` and compare errors.
- **Ex 5:** before running it, predict how each model does on the three test sets as training data
  grows. The results are a concrete argument for building physics into the inputs of a surrogate
  instead of hoping the network discovers it.

## Blank-file challenge (definition of done)

Close every file. In 30 minutes, in a new file, with no references:
train an MLP to fit `f(x, y) = sin(pi x) cos(pi y)` on `[-1, 1]^2` from 2,000 random samples,
with a train/val split, normalization fitted on train only, early stopping, a saved best
checkpoint, and a reload that reproduces the validation error. Target: < 1% relative L2.

## Common bugs

- Missing `optimizer.zero_grad()` (gradients accumulate across steps)
- Logging `loss` instead of `loss.item()` (keeps every graph alive; memory grows)
- Fitting the normalizer on validation or test data (leakage; errors look better than they are)
- Forgetting `model.eval()` / `torch.no_grad()` in evaluation
- Mixing devices (`Expected all tensors to be on the same device`) or dtypes (float64 NumPy -> float32 model)
- Selecting the best epoch on the test set instead of the validation set

## Time log

| Exercise | Planned | Actual | Notes |
|---|---|---|---|
| 1 | 1-1.5 h | | |
| 2 | 0.5-1 h | | |
| 3 | 1-1.5 h | | |
| 4 | 2-3 h | | |
| 5 | 1.5-2 h | | |
| Challenge | 0.5 h | | |
