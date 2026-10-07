"""
Paper tables: accuracy per variant (mean +- std over seeds) and paired bootstrap tests.

    python scripts/aggregate.py                                   # dev, best epoch of every runs/*_s<seed>
    python scripts/aggregate.py --ref dense16 --extra base=runs/b0_512/dev_preds.jsonl
    python scripts/aggregate.py --preds test_preds.jsonl          # files inside each run dir (held-out test)

Variant = run folder name without the _s<seed> suffix. Per run the predictions are
<run>/dev_preds_ep<best_epoch>.jsonl (from summary.json) unless --preds names a file.
Bootstrap (paired, stratified by task, uids common to both variants): resample uids within each
task, macro = mean of task means, per-uid score averaged over seeds. Reports the macro difference,
its 95% CI and p = P(diff <= 0) (one-sided, small = variant better than ref).
Writes runs/results/{table.md, table.tex, results.json}.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

TASKS = ("chart_qa", "document_ocr", "spatial_reasoning")


def read_preds(path):
    with open(path) as f:
        return {r["uid"]: r for r in map(json.loads, f) if r}


def collect(runs_dir, preds_name):
    """{variant: {seed: preds dict}}"""
    out = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(runs_dir, "*_s*"))):
        m = re.fullmatch(r"(.+)_s(\d+)", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        if preds_name:
            p = os.path.join(d, preds_name)
        else:
            sp = os.path.join(d, "summary.json")
            if not os.path.exists(sp):
                continue
            best = json.load(open(sp)).get("best_epoch")
            p = os.path.join(d, f"dev_preds_ep{best}.jsonl")
        if os.path.exists(p):
            out[m.group(1)][int(m.group(2))] = read_preds(p)
    return out


def task_scores(preds):
    by = defaultdict(list)
    for r in preds.values():
        by[r["task"]].append(r["score"])
    return {t: float(np.mean(v)) for t, v in by.items()}


def summarize(seeds):
    per_seed = [task_scores(p) for p in seeds.values()]
    tasks = sorted(per_seed[0])
    row = {"seeds": sorted(seeds)}
    for t in tasks:
        v = [s[t] for s in per_seed]
        row[t] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    macros = [np.mean([s[t] for t in tasks]) for s in per_seed]
    row["macro"] = (float(np.mean(macros)), float(np.std(macros, ddof=1)) if len(macros) > 1 else 0.0)
    return row


def uid_means(seeds):
    acc = defaultdict(list)
    task = {}
    for p in seeds.values():
        for u, r in p.items():
            acc[u].append(r["score"])
            task[u] = r["task"]
    return {u: float(np.mean(v)) for u, v in acc.items()}, task


def paired_bootstrap(a_seeds, b_seeds, n_boot=10000, seed=0):
    a, task = uid_means(a_seeds)
    b, _ = uid_means(b_seeds)
    uids = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for u in uids:
        groups[task[u]].append(u)
    diff_obs, boots = [], np.zeros(n_boot)
    for t, us in sorted(groups.items()):
        d = np.array([a[u] - b[u] for u in us])
        diff_obs.append(d.mean())
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots += d[idx].mean(1)
    boots /= len(groups)
    return {"diff": float(np.mean(diff_obs)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_le_0": float((boots <= 0).mean()), "n_uids": len(uids)}


def fmt(ms):
    return f"{ms[0]*100:.1f} ± {ms[1]*100:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--preds", default=None, help="predictions file name inside each run dir (default: best dev epoch)")
    ap.add_argument("--extra", nargs="*", default=[], help="name=path.jsonl for single runs (e.g. the base model)")
    ap.add_argument("--ref", default="dense16", help="variant the others are tested against")
    ap.add_argument("--out", default="runs/results")
    args = ap.parse_args()

    data = collect(args.runs, args.preds)
    for e in args.extra:
        name, path = e.split("=", 1)
        data[name] = {0: read_preds(path)}
    if not data:
        raise SystemExit(f"no runs with predictions under {args.runs}")
    rows = {v: summarize(s) for v, s in sorted(data.items())}
    tests = {v: paired_bootstrap(data[v], data[args.ref]) for v in data if v != args.ref and args.ref in data}

    tasks = [t for t in TASKS if any(t in r for r in rows.values())]
    head = ["variant", "seeds"] + tasks + ["macro", f"Δ vs {args.ref} [95% CI]", "p"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{tabular}{l" + "c" * (len(head) - 1) + "}", r"\toprule",
           " & ".join(h.replace("_", r"\_").replace("±", r"$\pm$").replace("Δ", r"$\Delta$") for h in head) + r" \\", r"\midrule"]
    for v, r in rows.items():
        t = tests.get(v)
        cells = [v, str(len(r["seeds"]))] + [fmt(r[x]) if x in r else "–" for x in tasks] + [fmt(r["macro"])]
        cells += ([f"{t['diff']*100:+.1f} [{t['ci95'][0]*100:+.1f}, {t['ci95'][1]*100:+.1f}]", f"{t['p_le_0']:.3f}"]
                  if t else ["–", "–"])
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(c.replace("_", r"\_").replace("±", r"$\pm$") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(args.out, "table.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    json.dump({"rows": rows, "tests": tests, "ref": args.ref, "preds": args.preds},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("\n".join(md))
    print(f"\n-> {args.out}/table.md, table.tex, results.json")


if __name__ == "__main__":
    main()
