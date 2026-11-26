"""Checkpointing schedules for a chain, a memory simulator, and the optimal schedule by DP.

A chain has activations x_0 … x_L (x_L is the loss) and layers 1 … L. A schedule is a list
of four operations:

- ``("fwd", l)``: run layer l without autograd on x_{l-1}, giving x_l;
- ``("ftape", l)``: run layer l with autograd, giving x_l and the layer's *tape*, the
  tensors autograd saves for backward. The tape also keeps x_{l-1} (the input) and x_l
  (the graph's root) alive until the layer's backward;
- ``("free", l)``: drop the schedule's reference to x_l;
- ``("bwd", l)``: run layer l's backward from g_l (the gradient of x_l), giving g_{l-1}
  and the parameter gradients, then free the tape and g_l.

Memory is counted in bytes over the activations that are alive, the tapes, and the
activation gradients (parameters and their gradients are a constant and not counted):

    held x_k (or kept by a tape) · a_k   +   live tapes · t_l   +   live gradients · g_k

During a backward, the tape, g_l and the new g_{l-1} are all alive at once; this is a
separate event in the trace. :func:`simulate` replays a schedule under this model, and
:class:`ckpt.executor.Executor` measures the same quantities on real tensors.

The optimal schedule is a recursion over sub-chains. ``solve(i, j, m)`` is the cheapest
way to turn {x_i, g_j} into {x_i, g_i} with peak memory at most m (x_i and the gradients
included). There are two ways to start:

- **tape** the next layer: ftape i+1, solve(i+1, j) with what is left, then bwd i+1;
- **checkpoint** at k: run layers i+1 … k without autograd, keep x_k, solve(k, j), free
  x_k, then solve(i, k) again from x_i.

Every schedule that stores activations and recomputes from stored ones is a tree of these
two choices, so the recursion is exact for this model. Taping everything is the
``store_all`` schedule; checkpointing at every layer is PyTorch's per-block
``checkpoint``; Chen et al.'s √L segments are one checkpoint per segment and tapes
inside. The memo is keyed on exact bytes by default. A chain has only a few distinct
sizes, so few distinct budgets m are ever reached. With a quantum q > 1, sizes are rounded
up, so any schedule the DP accepts still fits in exact bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache, lru_cache

INF = float("inf")


@dataclass
class Chain:
    """Sizes in bytes and times in seconds; index l is layer l (index 0 unused for layers)."""

    a: list  # a[k]: bytes of x_k, k = 0..L
    g: list  # g[k]: bytes of g_k (0 if x_k needs no gradient, as for token ids)
    t: list  # t[l]: tape bytes of layer l, beyond its input and output
    f: list  # f[l]: forward time without autograd
    ft: list  # ft[l]: forward time with autograd (taped)
    b: list  # b[l]: backward time
    names: list = field(default_factory=list)

    @property
    def L(self):
        return len(self.a) - 1


# -- schedules as trees ------------------------------------------------------------------
# ("B", i, j, sub): tape layer i+1, solve sub = (i+1, j) (None when j == i+1), bwd i+1.
# ("A", i, j, k, right, left): forward to x_k, solve right = (k, j), free x_k, solve left.


def emit(node, ops=None):
    ops = [] if ops is None else ops
    if node[0] == "B":
        _, i, j, sub = node
        ops.append(("ftape", i + 1))
        if sub is not None:
            emit(sub, ops)
        ops += [("free", i + 1), ("bwd", i + 1)]
    else:
        _, i, j, k, right, left = node
        for l in range(i + 1, k + 1):
            ops.append(("fwd", l))
            if l - 1 > i:
                ops.append(("free", l - 1))
        emit(right, ops)
        ops.append(("free", k))
        emit(left, ops)
    return ops


def store_all(i, j):
    """Tape every layer: plain backpropagation."""
    return ("B", i, j, store_all(i + 1, j) if j > i + 1 else None)


def per_layer(i, j):
    """Keep every layer's input, recompute each layer once (PyTorch's per-block checkpoint)."""
    if j == i + 1:
        return ("B", i, j, None)
    return ("A", i, j, i + 1, per_layer(i + 1, j), ("B", i, i + 1, None))


def segments(L, bounds):
    """Keep x at the given boundaries; tape everything inside each segment (Chen et al.)."""
    bounds = sorted(b for b in set(bounds) if 0 < b < L)
    starts = [0, *bounds]

    def build(n):  # from x_{starts[n]} to the loss; one forward pass stores every boundary
        i = starts[n]
        if n + 1 == len(starts):
            return store_all(i, L)
        k = starts[n + 1]
        return ("A", i, L, k, build(n + 1), store_all(i, k))

    return build(0)


def chen(L, s=None):
    """√L segments: checkpoints every s layers, s = ⌈√L⌉ by default."""
    s = s or math.ceil(math.sqrt(L))
    return segments(L, range(s, L, s))


# -- the memory model --------------------------------------------------------------------
def simulate(c: Chain, ops):
    """Replay ``ops`` from {x_0, g_L}. Returns (peak bytes, cost seconds, memory trace)."""
    L = c.L
    held, tapes, grads = {0}, set(), {L}
    done = set()
    trace = []

    def mem():
        alive = set(held)
        for l in tapes:
            alive |= {l - 1, l}
        return sum(c.a[k] for k in alive) + sum(c.t[l] for l in tapes) + sum(c.g[k] for k in grads)

    cost = 0.0
    for op, l in ops:
        if op in ("fwd", "ftape"):
            assert l - 1 in held, f"{op} {l}: x_{l - 1} is not held"
            assert l not in tapes
            held.add(l)
            if op == "ftape":
                tapes.add(l)
                cost += c.ft[l]
            else:
                cost += c.f[l]
        elif op == "free":
            assert l in held, f"free {l}: not held"
            held.discard(l)
        elif op == "bwd":
            assert l in tapes and l in grads, f"bwd {l}: needs the tape and g_{l}"
            assert l not in done
            if l > 1:
                grads.add(l - 1)
            trace.append(mem())  # tape, g_l and g_{l-1} together
            tapes.discard(l)
            grads.discard(l)
            done.add(l)
            cost += c.b[l]
        trace.append(mem())
    assert done == set(range(1, L + 1)), "every layer needs exactly one backward"
    assert not tapes and held == {0}
    return max(trace), cost, trace


# -- the optimal schedule ----------------------------------------------------------------
class Optimal:
    """Cheapest schedule under a memory budget, over all checkpoint/tape trees.

    ``allow_tape=False`` restricts the tape choice to single layers (every backward
    recomputes from a stored checkpoint), which is the setting of Griewank's ``revolve``.
    """

    def __init__(self, c: Chain, q=None, allow_tape=True):
        self.c, self.allow_tape = c, allow_tape
        self.q = q or 1  # exact bytes: a chain has few distinct sizes, so few distinct m
        up = lambda x: -(-int(x) // self.q)
        self.A, self.T, self.G = [up(x) for x in c.a], [up(x) for x in c.t], [up(x) for x in c.g]
        self.solve = lru_cache(maxsize=None)(self._solve)

    def _solve(self, i, j, m):
        """(cost, choice) for {x_i, g_j} → {x_i, g_i} within m units; cost INF if impossible."""
        if m <= 0:
            return INF, None
        c, A, T, G = self.c, self.A, self.T, self.G
        best = (INF, None)
        # tape layer i+1
        held = A[i] + T[i + 1] + A[i + 1]
        if held + G[j] <= m and held + G[i + 1] + G[i] <= m:
            if j == i + 1:
                best = (c.ft[i + 1] + c.b[i + 1], ("B",))
            elif self.allow_tape:
                sub = self.solve(i + 1, j, m - A[i] - T[i + 1])[0]
                best = (c.ft[i + 1] + sub + c.b[i + 1], ("B",))
        # checkpoint at k
        fsum = 0.0
        for k in range(i + 1, j):
            fsum += c.f[k]
            if A[i] + (A[k - 1] if k - 1 > i else 0) + A[k] + G[j] > m:
                break  # the forward to any later k passes through this state too
            right = self.solve(k, j, m - A[i])[0]
            if right == INF:
                continue
            total = fsum + right + self.solve(i, k, m)[0]
            if total < best[0]:
                best = (total, ("A", k))
        return best

    def tree(self, budget, i=0, j=None):
        """The optimal tree for a budget in bytes, or None if no schedule fits."""
        j = self.c.L if j is None else j
        m = int(budget) // self.q
        if self.solve(i, j, m)[0] == INF:
            return None
        return self._build(i, j, m)

    def _build(self, i, j, m):
        cost, ch = self.solve(i, j, m)
        assert cost < INF
        if ch[0] == "B":
            if j == i + 1:
                return ("B", i, j, None)
            return ("B", i, j, self._build(i + 1, j, m - self.A[i] - self.T[i + 1]))
        k = ch[1]
        return ("A", i, j, k, self._build(k, j, m - self.A[i]), self._build(i, k, m))

    def cost(self, budget):
        return self.solve(0, self.c.L, int(budget) // self.q)[0]

    def min_budget(self, max_cost=INF):
        """Smallest budget (bytes, a multiple of q) with a schedule costing ≤ max_cost.

        The optimal cost never rises with the budget, so this is a binary search."""
        lo, hi = 1, sum(self.A) + sum(self.T) + sum(self.G) + 1
        while lo < hi:
            mid = (lo + hi) // 2
            cost = self.solve(0, self.c.L, mid)[0]
            if cost < INF and cost <= max_cost:
                hi = mid
            else:
                lo = mid + 1
        return lo * self.q


# -- Griewank's binomial checkpointing ---------------------------------------------------
def revolve_forwards(l, s):
    """Fewest forward steps to reverse l steps with s checkpoint slots (one holds x_0)."""

    @cache
    def p(l, s):
        if l == 1:
            return 0
        if s == 1:
            return l * (l - 1) // 2
        return min(k + p(l - k, s - 1) + p(k, s) for k in range(1, l))

    return p(l, s)


def griewank(l, s):
    """Closed form of :func:`revolve_forwards`: t·l − C(s+t, s+1), β(s, t−1) < l ≤ β(s, t)."""
    beta = lambda s, t: math.comb(s + t, s)
    t = 0
    while beta(s, t) < l:
        t += 1
    return t * l - beta(s + 1, t - 1)
