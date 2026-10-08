#!/usr/bin/env bash
# Efficiency experiment (Kaggle commit, 2x T4), all on the held-out TEST split:
#   * reference: the original fp16 model (no quantization) at 512 px, with weights + peak VRAM
#   * 4-bit base at 256 / 512 px
#   * moe_seq and dense16 trained at 256 px (3 seeds each; teacher cache rebuilt at 256 px)
# Question answered: does 4-bit + adapters at 256 px match the fp16 model at 512 px with ~3x less
# memory and ~45% fewer image tokens?   Resumable; results in runs/results_eff/SUMMARY.txt
cd "$(dirname "$0")/.."
mkdir -p logs runs/results_eff
say() { echo "[$(date +%H:%M:%S)] $*"; }
T=data/v3/test.jsonl

base_eval() {  # gpu name extra-args...
    local gpu=$1 name=$2; shift 2
    [ -f runs/$name/test_preds.jsonl ] && return
    say "GPU$gpu $name"
    env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --base --split $T --out runs/$name/test_preds.jsonl "$@" \
        > logs/eff_$name.log 2>&1 || say "!! $name failed"
}
train_eval() { # gpu config seed
    local gpu=$1 cfg=$2 seed=$3 r=runs/$2_s$3
    if [ ! -f $r/summary.json ]; then
        say "GPU$gpu train $r"
        env CUDA_VISIBLE_DEVICES=$gpu python train.py --config configs/$cfg.yaml --seed $seed > logs/${cfg}_s${seed}.log 2>&1
        grep -E "^done" logs/${cfg}_s${seed}.log || say "!! $r failed"
    fi
    [ -f $r/test_preds.jsonl ] || env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --config configs/$cfg.yaml \
        --adapters $r/best --split $T --out $r/test_preds.jsonl > logs/eff_test_$2_s$3.log 2>&1
}

# 1. references + 256-px teacher cache (cache needs both GPUs; references squeeze in alongside)
(
    base_eval 0 fp16_512 --fp16
    CUDA_VISIBLE_DEVICES=0 python scripts/cache_teacher.py --config configs/moe_seq_256.yaml --shard 0 --num_shards 2 > logs/cache256_0.log 2>&1
) &
(
    base_eval 1 b0_256 --max_pixels 200704
    base_eval 1 b0_512
    CUDA_VISIBLE_DEVICES=1 python scripts/cache_teacher.py --config configs/moe_seq_256.yaml --shard 1 --num_shards 2 > logs/cache256_1.log 2>&1
) &
wait
python scripts/cache_teacher.py --config configs/moe_seq_256.yaml --check | tail -n 3
python scripts/cache_teacher.py --config configs/moe_seq_256.yaml --check > /dev/null 2>&1 || { say "!! 256-px cache incomplete"; exit 1; }

# 2. training at 256 px, one queue per GPU
( train_eval 0 moe_seq_256 0; train_eval 0 moe_seq_256 1; train_eval 0 moe_seq_256 2 ) &
( train_eval 1 dense16_256 0; train_eval 1 dense16_256 1; train_eval 1 dense16_256 2 ) &
wait

# 3. summary
S=runs/results_eff/SUMMARY.txt
{
    echo "== references on TEST (score / image tokens / latency, then memory)"
    for n in fp16_512 b0_512 b0_256; do echo "-- $n"; grep -E "^macro|weights" logs/eff_$n.log; done
    echo; echo "== 256-px runs (dev best, then TEST)"
    grep -hE "^done" logs/moe_seq_256_s*.log logs/dense16_256_s*.log
    for f in logs/eff_test_*.log; do echo "-- $f"; grep -E "^macro|weights" $f; done
    echo; echo "== TEST table at 256 px (Δ vs the fp16 model at 512 px, paired bootstrap)"
    python scripts/aggregate.py --runs runs --preds test_preds.jsonl --ref fp16_512 \
        --extra fp16_512=runs/fp16_512/test_preds.jsonl b0_512=runs/b0_512/test_preds.jsonl \
                b0_256=runs/b0_256/test_preds.jsonl --out runs/results_eff
} > $S 2>&1
cat $S
say "done -> $S"
