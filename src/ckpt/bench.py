"""Memory and time of checkpointing schedules on a GPT, predicted and measured.

    python -m ckpt.bench --out results

For each schedule this records the peak the model predicts, the peak the executor
measures on real storages, the peak resident memory above the pre-run level (from
``/proc/self/statm``, which also sees autograd's temporaries), and the predicted and
measured time. glibc's dynamic mmap threshold would keep freed blocks resident, so the
benchmark re-executes itself with ``MALLOC_MMAP_THRESHOLD_=65536`` on Linux.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch

from .executor import Executor
from .model import GPT, GPTConfig
from .profile import profile
from .schedule import Optimal, chen, emit, per_layer, simulate, store_all


def _rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class RSSPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = _rss(), False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, _rss())
            time.sleep(0.0005)


def measure(model, ex, ops, x0, reps):
    """(executor peak, Δ peak RSS, median wall time over reps) for one schedule."""
    trace, _ = ex.run(ops, x0, measure=True)
    model.zero_grad(set_to_none=True)
    rss = None
    if os.path.exists("/proc/self/statm"):
        for p in model.parameters():  # gradient buffers exist before the run
            p.grad = torch.zeros_like(p)
        before = _rss()
        s = RSSPeak()
        s.start()
        ex.run(ops, x0, measure=False)
        s.stop = True
        s.join()
        rss = s.peak - before
    times = []
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        times.append(ex.run(ops, x0, measure=False)[1])
    model.zero_grad(set_to_none=True)
    return max(trace), rss, statistics.median(times) if times else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--points", type=int, default=10)
    a = ap.parse_args(argv)
    if sys.platform == "linux" and "MALLOC_MMAP_THRESHOLD_" not in os.environ:
        env = dict(os.environ, MALLOC_MMAP_THRESHOLD_="65536")
        os.execve(sys.executable, [sys.executable, "-m", "ckpt.bench", *sys.argv[1:]], env)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=a.vocab, ctx=a.ctx, d=a.d, blocks=a.blocks, heads=a.heads)
    model = GPT(cfg)
    idx = torch.randint(0, a.vocab, (a.batch, a.ctx + 1))
    x0, tg = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    model.set_targets(tg)
    ch = profile(model.layers, x0, exclude=[tg], reps=a.reps)
    ex = Executor(model.layers, exclude=[tg])
    L = ch.L
    opt = Optimal(ch)
    lo = opt.min_budget()
    hi = simulate(ch, emit(store_all(0, L)))[0]

    scheds = []
    for name, tree in [("store all", store_all(0, L)), ("per layer", per_layer(0, L)),
                       ("√L segments", chen(L))]:  # fmt: skip
        ops = emit(tree)
        scheds.append((name, ops))
        cost = simulate(ch, ops)[1]
        # the least memory at which the optimal schedule is no slower (in the model)
        b = opt.min_budget(max_cost=cost * (1 + 1e-9))
        scheds.append((f"optimal, same compute as {name}", emit(opt.tree(b))))
    for n in range(a.points):
        budget = lo + (hi - lo) * n // (a.points - 1)
        scheds.append((f"optimal within {budget / 1e6:.1f} MB", emit(opt.tree(budget))))

    rows = []
    for name, ops in scheds:
        peak, cost, _ = simulate(ch, ops)
        t0 = time.perf_counter()
        got, rss, _ = measure(model, ex, ops, x0, 0)
        print(f"measured {name}: {len(ops)} ops in {time.perf_counter() - t0:.1f}s", flush=True)
        rows.append({
            "schedule": name, "predicted_peak": peak, "measured_peak": got, "rss_peak": rss,
            "predicted_s": cost, "forwards": sum(op == "fwd" for op, _ in ops) + L,
            "times": [],
        })  # fmt: skip
    for _ in range(a.reps):  # round-robin, so drift on a shared machine hits all alike
        for r, (_, ops) in zip(rows, scheds, strict=True):
            model.zero_grad(set_to_none=True)
            r["times"].append(ex.run(ops, x0, measure=False)[1])
    model.zero_grad(set_to_none=True)
    for r in rows:
        r["measured_s"] = statistics.median(r["times"])
        print(json.dumps(r), flush=True)

    with open(os.path.join(a.out, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "chain.json"), "w") as f:
        json.dump(ch.__dict__, f)
    params = sum(p.numel() for p in model.parameters())
    tbl = [f"GPT with {params / 1e6:.1f}M parameters: d = {a.d}, {a.blocks} blocks, "
           f"vocabulary {a.vocab}, batch {a.batch} × {a.ctx} tokens, fp32, "
           f"{torch.get_num_threads()} CPU threads. Chain: x_0 → embed → {a.blocks} blocks "
           f"→ head + loss, L = {L}. Memory is activation memory (parameters excluded).", "",
           "| schedule | predicted peak MB | measured peak MB | Δ RSS peak MB | forwards "
           "| predicted s | measured s |", "|---" * 7 + "|"]  # fmt: skip
    for r in rows:
        rss = "" if r["rss_peak"] is None else f"{r['rss_peak'] / 1e6:.1f}"
        tbl.append(
            f"| {r['schedule']} | {r['predicted_peak'] / 1e6:.2f} | "
            f"{r['measured_peak'] / 1e6:.2f} | {rss} | {r['forwards']} | "
            f"{r['predicted_s']:.3f} | {r['measured_s']:.3f} |"
        )
    prof = ["", "| layer | x MB | tape MB | forward ms | taped forward ms | backward ms |",
            "|---" * 6 + "|"]  # fmt: skip
    for l in range(1, L + 1):
        if 2 < l < L - 1:
            continue
        ms = " | ".join(f"{v[l] * 1e3:.1f}" for v in (ch.f, ch.ft, ch.b))
        prof.append(f"| {l}: {ch.names[l]} | {ch.a[l] / 1e6:.2f} | {ch.t[l] / 1e6:.2f} | {ms} |")
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(tbl + prof) + "\n")
    print("\n".join(tbl + prof))


if __name__ == "__main__":
    main()
