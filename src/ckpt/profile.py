"""Measure a chain: activation, gradient and tape bytes, and per-layer times."""

from __future__ import annotations

import statistics
import time

import torch
from torch.autograd.graph import saved_tensors_hooks

from .executor import tape_bytes
from .schedule import Chain


def _nbytes(t):
    return t.numel() * t.element_size()


def _time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def profile(layers, x0, exclude=(), reps=5):
    layers = list(layers)
    ex = {t.untyped_storage().data_ptr() for f in layers for t in f.parameters()}
    ex |= {t.untyped_storage().data_ptr() for t in exclude}
    a, g, t, f, ft, b = [_nbytes(x0)], [0], [0], [0.0], [0.0], [0.0]
    x = x0
    for layer in layers:
        tb, y = tape_bytes(layer, x, ex)
        a.append(_nbytes(y))
        g.append(_nbytes(y))
        t.append(tb)

        def fwd(layer=layer, x=x):
            with torch.no_grad():
                layer(x)

        def taped(layer=layer, x=x):
            xin = x.detach().requires_grad_() if x.is_floating_point() else x
            saved = []

            def pack(s):
                saved.append(s)
                return len(saved) - 1

            with saved_tensors_hooks(pack, lambda i: saved[i]):
                y = layer(xin)
            return xin, y

        gy = torch.ones_like(y)

        def bwd(layer=layer, x=x, gy=gy):
            xin, y = taped(layer, x)
            t0 = time.perf_counter()
            torch.autograd.backward(y, gy)
            return time.perf_counter() - t0

        fwd()  # warm-ups: the first backward of a layer allocates its weight gradients
        bwd()
        bwd()
        f.append(_time(fwd, reps))
        ft.append(_time(taped, reps))
        b.append(statistics.median(bwd() for _ in range(reps)))
        x = y
    if x0.is_floating_point():  # the executor then produces g_0 as well
        g[0] = a[0]
    for layer in layers:
        layer.zero_grad(set_to_none=True)
    names = [type(layer).__name__ for layer in layers]
    return Chain(a=a, g=g, t=t, f=f, ft=ft, b=b, names=["input", *names])
