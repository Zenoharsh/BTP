"""Tests for scripts/aggregate.py on synthetic prediction files.   python test_aggregate.py"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from scripts.aggregate import collect, paired_bootstrap, summarize

TASKS = ["chart_qa", "document_ocr", "spatial_reasoning"]


def write_run(root, name, seed, p_correct, n=50, best=1):
    d = os.path.join(root, f"{name}_s{seed}")
    os.makedirs(d)
    rng = np.random.default_rng([seed, int(p_correct * 100), len(name)])
    json.dump({"best_epoch": best}, open(os.path.join(d, "summary.json"), "w"))
    with open(os.path.join(d, f"dev_preds_ep{best}.jsonl"), "w") as f:
        for t in TASKS:
            for i in range(n):
                f.write(json.dumps({"uid": f"{t}_{i}", "task": t, "score": float(rng.random() < p_correct)}) + "\n")


def test_tables_and_bootstrap():
    with tempfile.TemporaryDirectory() as d:
        for s in range(3):
            write_run(d, "dense16", s, 0.6)
            write_run(d, "moe_seq", s, 0.9)
            write_run(d, "moe", s, 0.6)
        data = collect(d, None)
        assert set(data) == {"dense16", "moe_seq", "moe"} and sorted(data["moe"]) == [0, 1, 2]
        r = summarize(data["moe_seq"])
        assert abs(r["macro"][0] - 0.9) < 0.06 and r["macro"][1] > 0
        better = paired_bootstrap(data["moe_seq"], data["dense16"])
        same = paired_bootstrap(data["moe"], data["dense16"])
        assert better["diff"] > 0.2 and better["p_le_0"] < 0.001 and better["ci95"][0] > 0
        assert same["ci95"][0] < 0 < same["ci95"][1] and better["n_uids"] == 150
        out = os.path.join(d, "res")
        subprocess.run([sys.executable, "scripts/aggregate.py", "--runs", d, "--out", out], check=True,
                       capture_output=True)
        md = open(os.path.join(out, "table.md"), encoding="utf-8").read()
        assert "moe_seq" in md and "Δ vs dense16" in md
        assert os.path.exists(os.path.join(out, "table.tex"))


if __name__ == "__main__":
    test_tables_and_bootstrap()
    print("[PASS] test_tables_and_bootstrap\nAll 1 aggregate tests passed.")
