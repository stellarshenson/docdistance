"""Readings behind the docstrings of ``_exact_ot`` and ``_emd_iter_cap`` (DEF-TRANS-9).

Reads no file and loads no model: every problem is built from seeded random unit vectors and
solved with POT. It prints the result POT returns when the solver stops at an iteration cap (a
12 x 9 problem after 1 iteration, a 3000 x 3000 problem at POT's default cap), the smallest cap of
a coarse grid at which the 3000 x 3000 problem is solved, whether an ``n x 1`` problem is solved
at ``n - 1`` and at ``n`` iterations, and the iterations a 60000 x 2 problem needs.

Run from the repository root:

    CI=true .venv/bin/python reports/E13-campaign/checks/solver-cap/cap_readings.py
"""

import inspect
import warnings

import numpy as np
import ot

from docdistance import distance as d


def emb(n, dim=32, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def solve(a, b, C, cap):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan, log = ot.emd(a, b, C, numItermax=cap, log=True)
    return plan, log


if __name__ == "__main__":
    default_cap = inspect.signature(ot.emd).parameters["numItermax"].default
    print("POT", ot.__version__, "| default numItermax", default_cap)
    # the fixture of the raise test, stopped after 1 iteration
    X, Y = emb(12, seed=23), emb(9, seed=24)
    a, b, C = d._uniform(12), d._uniform(9), d.cost_matrix(X, Y)
    plan, log = solve(a, b, C, 1)
    best = ot.emd2(a, b, C)
    print(
        f"12 x 9 at cap 1: warning {log['warning']!r} | plan mass {plan.sum():.6f}"
        f" | row error {np.abs(plan.sum(1) - a).max():.2e}"
        f" | column error {np.abs(plan.sum(0) - b).max():.2e}"
        f" | cost {log['cost']:.6f} | optimum {best:.6f}"
    )
    # the 3000 x 3000 fixture of the cap test at POT's default cap
    x = np.random.default_rng(0).random((6000, 768)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    A, B = x[:3000], x[3000:]
    a, b, C = d._uniform(3000), d._uniform(3000), d.cost_matrix(A, B)
    plan, log = solve(a, b, C, 100_000)
    print(
        f"3000 x 3000 at cap 100000: warning {log['warning'] is not None}"
        f" | row error {np.abs(plan.sum(1) - a).max():.2e}"
        f" | column error {np.abs(plan.sum(0) - b).max():.2e}"
        f" | cost {log['cost']:.6f} | optimum {d.smd(A, B):.6f}"
    )
    # iterations the 3000 x 3000 fixture needs: smallest cap of a coarse grid that gives no warning
    for cap in (130_000, 135_000, 137_000, 138_000, 139_000, 140_000):
        print(f"   cap {cap}: optimal {solve(a, b, C, cap)[1]['warning'] is None}")
    # n x 1 needs exactly n iterations
    for n in (500, 2000, 150_000):
        Xn = np.random.default_rng(1).random((n, 8)).astype(np.float32)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        a, b, C = d._uniform(n), d._uniform(1), d.cost_matrix(Xn, Xn[:1])
        below = solve(a, b, C, n - 1)[1]["warning"] is None
        at = solve(a, b, C, n)[1]["warning"] is None
        print(
            f"{n} x 1: optimal at cap n - 1 {below} | optimal at cap n {at}"
            f" | shipped cap {d._emd_iter_cap(n, 1)}"
        )
    # 60000 x 2: the reading the docstring quotes
    n = 60_000
    Xn = np.random.default_rng(2).random((n + 2, 8)).astype(np.float32)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    a, b, C = d._uniform(n), d._uniform(2), d.cost_matrix(Xn[:n], Xn[n:])
    lo, hi = 1, n * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if solve(a, b, C, mid)[1]["warning"] is None:
            hi = mid
        else:
            lo = mid + 1
    print(f"60000 x 2: needs {lo} iterations, {lo / (n * 2):.3f} per arc")
