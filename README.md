# activation-checkpointing-from-scratch

[![CI](https://github.com/kishorhange111/activation-checkpointing-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/activation-checkpointing-from-scratch/actions/workflows/ci.yml)

Activation checkpointing (rematerialisation) for transformer training, from first
principles:
- an exact memory model of a training step;
- a dynamic program that finds the fastest schedule within a memory budget, over every
  store-or-recompute schedule;
- an executor that runs any schedule on a real GPT.

The executor's measured memory equals the model **at every event of every schedule**, and
every schedule gives gradients bitwise identical to plain backpropagation, dropout
included. At the compute cost of PyTorch's per-block checkpointing, the optimal schedule
needs 18% less activation memory, and at the cost of Chen et al.'s √L segments, 17% less.

## The model ([`schedule.py`](src/ckpt/schedule.py))

The network is a chain x_0 → x_1 → … → x_L: token ids, an embedding, 16 transformer
blocks, and a head that returns the loss. A schedule is a sequence of four operations:
- `fwd l`: run layer l without autograd;
- `ftape l`: run layer l with autograd, keeping its *tape* (what autograd saves for
  backward), its input and its output;
- `free l`: drop x_l;
- `bwd l`: turn g_l into g_{l-1}, then free the tape.

Memory is the sum of the live activations, tapes and activation gradients. Parameters
and their gradients are constant and are left out.

`solve(i, j, m)` is the cheapest way to turn {x_i, g_j} into {x_i, g_i} with peak
memory ≤ m. It starts in one of two ways:
- **tape** layer i+1 and solve (i+1, j) in the memory that is left;
- **checkpoint** at k: run to x_k without autograd, solve (k, j), free x_k, then solve
  (i, k) from x_i again.

Every schedule of this kind is a tree of those two choices, so the recursion is exact.
Plain backprop, per-layer checkpointing and √L segments are three particular trees. The
memo is keyed on exact bytes. A chain has few distinct sizes, so the 18-layer GPT solves
in under a second.

Three checks that it is right:
- **Brute force.** On random heterogeneous chains, the DP's cost equals the minimum over
  *every* tree at every budget, and its tree meets the budget.
- **Griewank's binomial optimum.** With unit activations, free tapes and only
  forward steps counted, the DP gives exactly the forward count of `revolve` with s
  checkpoint slots at budget s + 2. The two extra units are a layer's input and output.
  This holds for s ≤ 4 and l ≤ 25. The recurrence also equals the closed form
  t·l − C(s+t, s+1) for l < 60, s ≤ 6.
- **Real tensors** (below).

## The executor ([`executor.py`](src/ckpt/executor.py))

A taped forward runs under `torch.autograd.graph.saved_tensors_hooks`. The pack hook
keeps each saved tensor in the layer's tape and hands autograd an index. So the tape is
exactly what autograd saved, and it dies when the executor drops it. Memory is the total
size of the unique storages the executor holds, measured after every operation and once
inside each backward.

The tests check:
- that trace equals `simulate`'s, element for element;
- with weak references, that every freed tape storage is really gone.

Dropout masks would differ on recomputation, so the executor records the RNG state the
first time each layer runs and restores it whenever the layer is recomputed. A test shows
that without this, gradients change.

## Results ([`bench.py`](src/ckpt/bench.py), [results/bench.md](results/bench.md))

The model is a 16.9M-parameter GPT: d = 256, 16 blocks, vocabulary 8192, a batch of 8 ×
256 tokens, fp32. It runs on 4 CPU threads in WSL2. A block's tape is 31.6 MB, against
a 2.1 MB block input. The head's tape is 69.2 MB, mostly the 67.1 MB log-softmax of the logits. Times are
medians of 7 interleaved runs.

| schedule | peak MB: model = measured | forwards | predicted s | measured s |
|---|---|---|---|---|
| store all (plain backprop) | 611.9 | 18 | 1.96 | 1.91 |
| per layer (PyTorch `checkpoint` per block) | 107.0 | 35 | 2.50 | 2.45 |
| **optimal, same compute as per layer** | **88.1** | 35 | 2.50 | 2.45 |
| √L segments (Chen et al.) | 176.7 | 33 | 2.43 | 2.38 |
| **optimal, same compute as √L** | **147.0** | 33 | 2.43 | 2.39 |
| optimal, least memory possible | 73.4 | 50 | 3.00 | 2.91 |

- **The memory model is exact.** In all 16 schedules benchmarked (the six above and a
  sweep of ten budgets in [bench.md](results/bench.md)), the executor's measured peak
  equals the predicted peak to the byte.
- **Optimal beats the heuristics at equal compute.** At the cost of per-layer
  checkpointing, the DP keeps the same number of forwards but places the checkpoints
  better: 88.1 MB instead of 107.0. At the cost of √L segments it needs 147.0 MB instead
  of 176.7.
- **The trade-off curve.** Activation memory can fall to 12% of plain backprop (73.4 MB)
  for 53% more step time (2.91 s vs 1.91 s). The ten-budget sweep traces the curve in
  between.
- **Predicted time** (a sum of per-layer profiled times) is within 5% of measured time in
  every schedule.
- **What the model does not count.** Peak resident memory (`/proc/self/statm`, with
  `MALLOC_MMAP_THRESHOLD_` fixed so freed blocks leave) is 100–132 MB above the model in
  every schedule. That is autograd's workspace inside a backward; the head's logits
  gradient alone is 67 MB. The model counts tensors held between operations. From per
  layer to its optimal counterpart, resident memory falls by 19 MB, as the model says.
  From √L segments to its optimal counterpart, it does not fall at all: there the
  resident peak is set by a backward's workspace, not by the held tensors the DP
  minimises. Adding each layer's backward workspace to the model is the natural next
  step.

## Tests

`pytest -q` runs 22 tests:
- the DP against brute force over every tree, 12 random chains;
- a coarser quantum still gives feasible schedules;
- the optimal cost never rises with the budget, and `min_budget` is tight;
- the three baselines are trees the DP considers;
- revolve's recurrence equals Griewank's closed form, and the DP reproduces revolve;
- `simulate` rejects invalid schedules;
- on a real GPT with dropout:
  - measured memory equals the model at every event;
  - gradients are bitwise identical across schedules and match plain autograd;
  - without RNG restoration, gradients differ;
  - freed tapes really die.

```
pip install -e ".[dev]" && pytest -q
python -m ckpt.bench --out results    # Linux also measures resident memory
```
