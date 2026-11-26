import random

import pytest
import torch

from ckpt.executor import Executor
from ckpt.model import GPT, GPTConfig
from ckpt.profile import profile
from ckpt.schedule import (
    INF,
    Chain,
    Optimal,
    chen,
    emit,
    griewank,
    per_layer,
    revolve_forwards,
    segments,
    simulate,
    store_all,
)

# -- the DP on abstract chains -----------------------------------------------------------


def all_trees(i, j):
    """Every checkpoint/tape tree for the sub-chain (i, j)."""
    if j == i + 1:
        yield ("B", i, j, None)
        return
    for sub in all_trees(i + 1, j):
        yield ("B", i, j, sub)
    for k in range(i + 1, j):
        for right in all_trees(k, j):
            for left in all_trees(i, k):
                yield ("A", i, j, k, right, left)


def random_chain(L, rng):
    a = [rng.randint(1, 4)] + [rng.randint(1, 9) for _ in range(L)]
    g = [0] + a[1:L] + [1]
    t = [0] + [rng.randint(0, 20) for _ in range(L)]
    f = [0.0] + [rng.randint(1, 5) for _ in range(L)]
    ft = [0.0] + [x + rng.randint(0, 2) for x in f[1:]]
    b = [0.0] + [rng.randint(1, 9) for _ in range(L)]
    return Chain(a=a, g=g, t=t, f=f, ft=ft, b=b)


@pytest.mark.parametrize("seed", range(12))
def test_dp_equals_brute_force_over_every_tree(seed):
    rng = random.Random(seed)
    c = random_chain(rng.randint(2, 6), rng)
    L = c.L
    sims = [simulate(c, emit(t))[:2] for t in all_trees(0, L)]
    opt = Optimal(c)
    for budget in sorted({p for p, _ in sims} | {min(p for p, _ in sims) - 1}):
        want = min((cost for p, cost in sims if p <= budget), default=INF)
        assert opt.cost(budget) == pytest.approx(want)
        tree = opt.tree(budget)
        if want == INF:
            assert tree is None
            continue
        peak, cost, _ = simulate(c, emit(tree))
        assert peak <= budget and cost == pytest.approx(want)


def test_quantized_dp_stays_feasible():
    rng = random.Random(7)
    for _ in range(20):
        c = random_chain(6, rng)
        exact = Optimal(c)
        coarse = Optimal(c, q=3)
        for budget in range(exact.min_budget(), 200, 7):
            tree = coarse.tree(budget)
            if tree is not None:
                peak, cost, _ = simulate(c, emit(tree))
                assert peak <= budget and cost >= exact.cost(budget) - 1e-9


def test_cost_never_rises_with_budget_and_min_budget_is_tight():
    c = random_chain(8, random.Random(3))
    opt = Optimal(c)
    lo = opt.min_budget()
    assert opt.cost(lo - 1) == INF and opt.cost(lo) < INF
    costs = [opt.cost(m) for m in range(lo, lo + 150)]
    assert all(x >= y for x, y in zip(costs, costs[1:], strict=False))
    target = simulate(c, emit(per_layer(0, c.L)))[1]
    m = opt.min_budget(max_cost=target)
    assert opt.cost(m) <= target < opt.cost(m - 1)


def test_baselines_are_trees_the_dp_considers():
    c = random_chain(9, random.Random(5))
    opt = Optimal(c)
    for tree in (store_all(0, 9), per_layer(0, 9), chen(9), segments(9, [2, 5, 7])):
        peak, cost, _ = simulate(c, emit(tree))
        assert opt.cost(peak) <= cost + 1e-9


def test_griewank_closed_form():
    for s in range(1, 7):
        for l in range(1, 60):
            assert revolve_forwards(l, s) == griewank(l, s)


def test_dp_reproduces_revolve():
    # unit activations, free tapes, only forwards cost: s checkpoint slots plus two
    # working activations (a layer's input and output) is exactly revolve with s slots
    for s in range(1, 5):
        for l in range(2, 26):
            c = Chain(a=[1] * (l + 1), g=[0] * (l + 1), t=[0] * (l + 1),
                      f=[0] + [1] * l, ft=[0] * (l + 1), b=[0] * (l + 1))  # fmt: skip
            assert Optimal(c, allow_tape=False).cost(s + 2) == revolve_forwards(l, s)


def test_simulate_rejects_invalid_schedules():
    c = random_chain(3, random.Random(0))
    with pytest.raises(AssertionError):
        simulate(c, [("bwd", 3)])
    with pytest.raises(AssertionError):
        simulate(c, [("fwd", 2)])
    ops = emit(store_all(0, 3))
    with pytest.raises(AssertionError):
        simulate(c, ops[:-1])


# -- real tensors ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = GPT(GPTConfig(blocks=5, dropout=0.1))
    idx = torch.randint(0, 256, (3, 65))
    x0, tg = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    model.set_targets(tg)
    chain = profile(model.layers, x0, exclude=[tg], reps=1)
    return model, x0, tg, chain


def schedules(chain):
    L, opt = chain.L, Optimal(chain)
    lo, hi = opt.min_budget(), simulate(chain, emit(store_all(0, L)))[0]
    out = {"store all": store_all(0, L), "per layer": per_layer(0, L), "chen": chen(L)}
    for n in range(4):
        out[f"optimal {n}"] = opt.tree(lo + (hi - lo) * n // 3)
    return {k: emit(v) for k, v in out.items()}


def grads(model, ex, ops, x0, seed=1):
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    trace, _ = ex.run(ops, x0)
    out = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    return out, trace


def test_measured_memory_equals_model_at_every_event(setup):
    model, x0, tg, chain = setup
    ex = Executor(model.layers, exclude=[tg])
    for name, ops in schedules(chain).items():
        _, trace = grads(model, ex, ops, x0)
        assert trace == simulate(chain, ops)[2], name


def test_every_schedule_gives_bitwise_identical_gradients(setup):
    model, x0, tg, chain = setup
    ex = Executor(model.layers, exclude=[tg])
    s = schedules(chain)
    ref, _ = grads(model, ex, s["store all"], x0)
    for name, ops in s.items():
        got, _ = grads(model, ex, ops, x0)
        assert all(torch.equal(a, b) for a, b in zip(ref, got, strict=True)), name
    # and store-all through the executor matches plain autograd on the whole model
    torch.manual_seed(1)
    model(x0, tg).backward()
    plain = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for a, b in zip(ref, plain, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_recomputation_without_rng_restore_changes_dropout(setup):
    model, x0, tg, chain = setup
    ops = schedules(chain)["per layer"]
    ref, _ = grads(model, Executor(model.layers, exclude=[tg]), ops, x0)
    bad, _ = grads(model, Executor(model.layers, exclude=[tg], preserve_rng=False), ops, x0)
    assert not all(torch.equal(a, b) for a, b in zip(ref, bad, strict=True))


def test_freed_tapes_really_die(setup):
    model, x0, tg, chain = setup
    ex = Executor(model.layers, exclude=[tg])
    grads(model, ex, schedules(chain)["optimal 1"], x0)
    assert ex.refs and all(r.expired() for r in ex.refs)
