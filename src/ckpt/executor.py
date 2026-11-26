"""Runs a schedule on real tensors and measures the memory it holds.

Each taped forward runs under ``saved_tensors_hooks`` whose pack hook keeps the saved
tensor in the layer's tape list and hands autograd an index, so the tape is exactly the
set of tensors autograd saved, owned by the executor and freed when the layer's backward
is done. Memory after every operation is the total size of the unique storages the
executor holds (activations, tapes, activation gradients), excluding parameters and the
targets. That is the quantity :func:`ckpt.schedule.simulate` predicts.

Recomputation is bit-exact, including dropout: the RNG state is recorded the first time
each layer runs and restored whenever the layer is recomputed.
"""

from __future__ import annotations

import time

import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.multiprocessing.reductions import StorageWeakRef


def _storages(tensors, exclude):
    seen = {}
    for t in tensors:
        s = t.untyped_storage()
        p = s.data_ptr()
        if p in exclude or s.nbytes() == 0:
            continue
        seen[p] = max(seen.get(p, 0), s.nbytes())
    return seen


class Executor:
    def __init__(self, layers, preserve_rng=True, exclude=()):
        """``layers[l - 1]`` is layer l; ``exclude`` are tensors not counted (e.g. targets)."""
        self.layers, self.preserve_rng = list(layers), preserve_rng
        ex = [p for f in self.layers for p in f.parameters()] + list(exclude)
        self.exclude = {t.untyped_storage().data_ptr() for t in ex}
        self.refs = []  # weak refs to every saved storage: evidence that freed tapes die

    def _call(self, l, x, rng):
        if not self.preserve_rng:
            return self.layers[l - 1](x)
        if l not in rng:
            rng[l] = torch.get_rng_state()
            return self.layers[l - 1](x)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng[l])
            return self.layers[l - 1](x)

    def memory(self, x, g, tapes):
        ts = list(x.values()) + list(g.values())
        for xin, y, saved in tapes.values():
            ts += [xin, y, *saved]
        return sum(_storages(ts, self.exclude).values())

    def run(self, ops, x0, measure=True):
        """Returns (memory after each event, or None if not measuring; wall seconds)."""
        L = len(self.layers)
        x, g, tapes, rng = {0: x0}, {L: torch.ones(())}, {}, {}
        trace = [] if measure else None
        t0 = time.perf_counter()
        for op, l in ops:
            if op == "fwd":
                with torch.no_grad():
                    x[l] = self._call(l, x[l - 1], rng)
            elif op == "ftape":
                inp = x[l - 1]
                xin = inp.detach().requires_grad_() if inp.is_floating_point() else inp
                saved = []

                def pack(t, saved=saved):
                    saved.append(t)
                    return len(saved) - 1

                with saved_tensors_hooks(pack, lambda i, saved=saved: saved[i]):
                    y = self._call(l, xin, rng)
                tapes[l] = (xin, y, saved)
                x[l] = y.detach()
                if measure:
                    ex = self.exclude | {xin.untyped_storage().data_ptr()}
                    self.refs += [
                        StorageWeakRef(s.untyped_storage())
                        for s in saved
                        if s.untyped_storage().data_ptr() not in ex
                    ]
            elif op == "free":
                del x[l]
            elif op == "bwd":
                xin, y, saved = tapes[l]
                torch.autograd.backward(y, g[l])
                if xin.requires_grad:
                    g[l - 1] = xin.grad
                if measure:
                    trace.append(self.memory(x, g, tapes))
                del tapes[l], g[l], xin, y, saved
            if measure:
                trace.append(self.memory(x, g, tapes))
        return trace, time.perf_counter() - t0


def tape_bytes(layer, x, exclude):
    """Bytes autograd saves for one layer beyond its input and output, and a check run."""
    xin = x.detach().requires_grad_() if x.is_floating_point() else x
    saved = []

    def pack(t):
        saved.append(t)
        return len(saved) - 1

    with saved_tensors_hooks(pack, lambda i: saved[i]):
        y = layer(xin)
    ex = set(exclude) | {xin.untyped_storage().data_ptr(), y.untyped_storage().data_ptr()}
    return sum(_storages(saved, ex).values()), y.detach()
