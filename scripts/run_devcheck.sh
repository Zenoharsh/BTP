#!/usr/bin/env bash
# Dev re-check (Kaggle commit, 2x T4): re-evaluate the saved best/ adapters of the HQQ runs on DEV with
# evaluate.py and compare with the dev score recorded during training (summary.json). Rules out a
# difference between the in-training evaluation and the reloaded model. One NF4 moe_seq run is
# re-evaluated as a control if its adapters are attached.
# Attach as Inputs: the run_mopeq_train output (HQQ runs + plans), optionally an output with
# runs/moe_seq_s0/best (the run_all version). Results in runs/results_devcheck/SUMMARY.txt
cd "$(dirname "$0")/.."
mkdir -p logs runs/mopeq runs/results_devcheck
say() { echo "[$(date +%H:%M:%S)] $*"; }
D=data/v3/dev.jsonl
pip install -q hqq

fetch() {  # path-suffix -> copies the first match under /kaggle/input to ./path-suffix
    [ -e "$1" ] && return
    local hit; hit=$(find /kaggle/input -path "*/$1" 2>/dev/null | head -n 1)
    [ -n "$hit" ] && { mkdir -p "$(dirname "$1")"; cp -r "$hit" "$1"; say "reused $1"; }
}
for p in plan_kl_3.5 plan_u4; do fetch runs/mopeq/$p.json; done
RUNS=""
for r in moe_seq_hqq4_s0 moe_seq_hqq4_s1 moe_seq_hqq4_s2 moe_seq_mopeq35_s0 moe_seq_mopeq35_s1 moe_seq_mopeq35_s2 moe_seq_s0; do
    fetch runs/$r/best; fetch runs/$r/summary.json
    [ -f runs/$r/best/adapters.pt ] && RUNS="$RUNS $r"
done
[ -n "$RUNS" ] || { say "!! no adapters found: attach the run_mopeq_train output"; exit 1; }
say "re-evaluating:$RUNS"

ev() {  # gpu run
    local gpu=$1 r=$2 cfg; cfg=$(echo $2 | sed -E 's/_s[0-9]+$//')
    [ -f runs/$r/dev_recheck_summary.json ] && return
    say "GPU$gpu dev $r"
    env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --config configs/$cfg.yaml --adapters runs/$r/best \
        --split $D --out runs/$r/dev_recheck.jsonl > logs/devcheck_$r.log 2>&1 || say "!! $r failed"
}
set -- $RUNS
( i=0; for r in "$@"; do [ $((i % 2)) -eq 0 ] && ev 0 $r; i=$((i + 1)); done ) &
( i=0; for r in "$@"; do [ $((i % 2)) -eq 1 ] && ev 1 $r; i=$((i + 1)); done ) &
wait

python - <<'PYEOF' | tee runs/results_devcheck/SUMMARY.txt
import glob, json, os
print(f"{'run':<22}{'dev in training':>17}{'dev re-eval':>13}{'diff':>8}   per task (re-eval)")
for f in sorted(glob.glob("runs/*/dev_recheck_summary.json")):
    r = os.path.basename(os.path.dirname(f))
    re_ = json.load(open(f))
    s = os.path.join("runs", r, "summary.json")
    tr = json.load(open(s))["best_dev_macro"] if os.path.exists(s) else float("nan")
    pt = " ".join(f"{t[:5]}={v['score']:.3f}" for t, v in re_["per_task"].items())
    print(f"{r:<22}{tr:>17.4f}{re_['macro']:>13.4f}{re_['macro'] - tr:>+8.4f}   {pt}")
PYEOF
say "done -> runs/results_devcheck/SUMMARY.txt"
