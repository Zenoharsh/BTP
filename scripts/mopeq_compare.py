"""Hessian (MoPEQ) vs KL sensitivity: Spearman rank agreement per bit-width and plan overlap per budget.

    python scripts/mopeq_compare.py [--dir runs/mopeq]
"""
import argparse
import glob
import json
import os


def load_sens(pattern):
    sens = {}
    for f in sorted(glob.glob(pattern)):
        sens.update(json.load(open(f))["sens"])
    return sens


def spearman(x, y):
    """Spearman rho of two {key: value} dicts over their common keys (average ranks for ties)."""
    keys = sorted(set(x) & set(y))

    def ranks(d):
        order = sorted(keys, key=lambda k: d[k])
        r, i = {}, 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and d[order[j + 1]] == d[order[i]]:
                j += 1
            for k in order[i:j + 1]:
                r[k] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(keys)
    mx, my = sum(rx.values()) / n, sum(ry.values()) / n
    cov = sum((rx[k] - mx) * (ry[k] - my) for k in keys)
    vx = sum((rx[k] - mx) ** 2 for k in keys) ** 0.5
    vy = sum((ry[k] - my) ** 2 for k in keys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan"), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/mopeq")
    args = ap.parse_args()
    kl = load_sens(os.path.join(args.dir, "sens_kl_[0-9].json"))
    he = load_sens(os.path.join(args.dir, "sens_hessian_[0-9].json"))
    if not kl or not he:
        print("need both KL and Hessian sensitivities")
        return
    for b in ("2", "3", "4"):
        rho, n = spearman({u: v[b] for u, v in kl.items()}, {u: v[b] for u, v in he.items()})
        print(f"Spearman rho KL vs Hessian at {b} bit over {n} units: {rho:.3f}")
    for a in ("3", "3.5", "4"):
        f1, f2 = (os.path.join(args.dir, f"plan_{m}_{a}.json") for m in ("kl", "hess"))
        if os.path.exists(f1) and os.path.exists(f2):
            p, q = json.load(open(f1))["bits"], json.load(open(f2))["bits"]
            same = sum(p[u] == q[u] for u in p)
            print(f"budget {a} bit: {same}/{len(p)} units get the same bits from both metrics")


if __name__ == "__main__":
    main()
