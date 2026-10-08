#!/usr/bin/env bash
# Held-out TEST evaluation of every trained run (Kaggle commit, 2x T4). Needs the output of the
# run_all.sh version attached as an Input (Add Input -> Your Work -> that notebook's version).
# Model selection used dev only; this touches test exactly once.
#
#   bash scripts/run_test.sh          -> runs/results_test/{SUMMARY.txt, table.md, table.tex}
cd "$(dirname "$0")/.."
mkdir -p logs runs
say() { echo "[$(date +%H:%M:%S)] $*"; }
T=data/v3/test.jsonl

if [ ! -d runs/moe_seq_s0/best ]; then
    hit=$(find /kaggle/input -path "*runs/moe_seq_s0/best/adapters.pt" 2>/dev/null | head -n 1)
    [ -n "$hit" ] || { say "!! no trained runs found under /kaggle/input (attach the run_all version as Input)"; exit 1; }
    src=$(dirname "$(dirname "$(dirname "$hit")")")
    say "copying runs from $src"
    cp -r "$src"/. runs/
fi

evalrun() {  # gpu run_dir
    local gpu=$1 r=$2 cfg; cfg=$(basename "$r" | sed -E 's/_s[0-9]+$//')
    [ -f "$r/test_preds.jsonl" ] && return
    say "GPU$gpu test $r"
    env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --config configs/$cfg.yaml --adapters $r/best --split $T \
        --out $r/test_preds.jsonl > logs/test_$(basename $r).log 2>&1 || say "!! $r failed"
}

runs=(runs/*_s[0-9])
(
    [ -f runs/b0_512/test_preds.jsonl ] || CUDA_VISIBLE_DEVICES=0 python evaluate.py --base --split $T \
        --out runs/b0_512/test_preds.jsonl > logs/test_b0.log 2>&1
    for ((i = 0; i < ${#runs[@]}; i += 2)); do evalrun 0 "${runs[$i]}"; done
) &
(
    for ((i = 1; i < ${#runs[@]}; i += 2)); do evalrun 1 "${runs[$i]}"; done
    for r in runs/moe_seq_s*; do     # deployable per-task checkpoints (profile from DEV routing)
        [ -f $r/convert_profile_test.json ] && continue
        CUDA_VISIBLE_DEVICES=1 python convert.py --config configs/moe_seq.yaml --adapters $r/best \
            --routing $r/routing_dev.npz --mode profile --eval --split $T --out $r/convert_profile_test.json \
            > logs/test_profile_$(basename $r).log 2>&1
    done
) &
wait

mkdir -p runs/results_test
S=runs/results_test/SUMMARY.txt
{
    echo "== TEST per run"; for f in logs/test_*.log; do echo "-- $f"; grep -E "^macro|profile:" $f; done
    echo; echo "== TEST 3-seed table (vs dense16, paired bootstrap)"
    python scripts/aggregate.py --preds test_preds.jsonl --ref dense16 \
        --extra base=runs/b0_512/test_preds.jsonl --out runs/results_test
} > $S 2>&1
cat $S
say "done -> $S"
