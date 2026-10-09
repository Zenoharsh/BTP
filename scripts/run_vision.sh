#!/usr/bin/env bash
# Mixed-precision experiment (Kaggle commit, 2x T4), all on the held-out TEST split, 512 px:
#   * where does the 4-bit loss come from? base model with only the vision encoder kept fp16
#     (b0vis_512) and with only the language model kept fp16 (b0lang_512), vs fp16_512 / b0_512
#   * moe_seq_vis and dense16_vis: the usual recipe with the vision encoder fp16, 3 seeds each
# Attach the outputs of the run_test (V2) and run_efficiency (V3) versions as Inputs to reuse the
# teacher cache, the fp16 / 4-bit references and the 4-bit moe_seq / dense16 test predictions;
# anything not found is recomputed.  Resumable; results in runs/results_vis/SUMMARY.txt
cd "$(dirname "$0")/.."
mkdir -p logs runs/results_vis
say() { echo "[$(date +%H:%M:%S)] $*"; }
T=data/v3/test.jsonl

# 0. reuse earlier outputs from /kaggle/input
fetch() {  # path-suffix  ->  copies the first match under /kaggle/input to ./path-suffix
    [ -e "$1" ] && return
    local hit; hit=$(find /kaggle/input -path "*/$1" 2>/dev/null | head -n 1)
    [ -n "$hit" ] && { mkdir -p "$(dirname "$1")"; cp -r "$hit" "$1"; say "reused $1"; }
}
fetch teacher_cache/v3
for n in fp16_512 b0_512; do fetch runs/$n/test_preds.jsonl; done
for r in moe_seq_s0 moe_seq_s1 moe_seq_s2 dense16_s0 dense16_s1 dense16_s2; do fetch runs/$r/test_preds.jsonl; done

base_eval() {  # gpu name extra-args...
    local gpu=$1 name=$2; shift 2
    [ -f runs/$name/test_preds.jsonl ] && return
    say "GPU$gpu $name"
    env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --base --split $T --out runs/$name/test_preds.jsonl "$@" \
        > logs/vis_$name.log 2>&1 || say "!! $name failed"
}
train_eval() { # gpu config seed
    local gpu=$1 cfg=$2 seed=$3 r=runs/$2_s$3
    if [ ! -f $r/summary.json ]; then
        say "GPU$gpu train $r"
        env CUDA_VISIBLE_DEVICES=$gpu python train.py --config configs/$cfg.yaml --seed $seed > logs/${cfg}_s${seed}.log 2>&1
        grep -E "^done" logs/${cfg}_s${seed}.log || say "!! $r failed"
    fi
    [ -f $r/test_preds.jsonl ] || env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --config configs/$cfg.yaml \
        --adapters $r/best --split $T --out $r/test_preds.jsonl > logs/vis_test_${cfg}_s${seed}.log 2>&1
}

# 1. base-model decomposition + teacher cache (only if it was not reused)
cache_ok() { python scripts/cache_teacher.py --check > /dev/null 2>&1; }
(
    base_eval 0 b0vis_512 --skip_quant vision
    base_eval 0 fp16_512 --fp16
    cache_ok || python scripts/cache_teacher.py --shard 0 --num_shards 2 > logs/cache_vis0.log 2>&1
) &
(
    base_eval 1 b0lang_512 --skip_quant language
    base_eval 1 b0_512
    cache_ok || CUDA_VISIBLE_DEVICES=1 python scripts/cache_teacher.py --shard 1 --num_shards 2 > logs/cache_vis1.log 2>&1
) &
wait
cache_ok || { say "!! teacher cache incomplete"; python scripts/cache_teacher.py --check | tail -n 3; exit 1; }

# 2. training with the vision encoder in fp16, one queue per GPU
( train_eval 0 moe_seq_vis 0; train_eval 0 moe_seq_vis 1; train_eval 0 moe_seq_vis 2 ) &
( train_eval 1 dense16_vis 0; train_eval 1 dense16_vis 1; train_eval 1 dense16_vis 2 ) &
wait

# 3. summary
S=runs/results_vis/SUMMARY.txt
X="fp16_512=runs/fp16_512/test_preds.jsonl b0_512=runs/b0_512/test_preds.jsonl
   b0vis_512=runs/b0vis_512/test_preds.jsonl b0lang_512=runs/b0lang_512/test_preds.jsonl"
{
    echo "== base model on TEST: which part loses accuracy in 4-bit (score / image tokens / latency, then memory)"
    for n in b0vis_512 b0lang_512 fp16_512 b0_512; do
        echo "-- $n"; grep -hE "^macro|^weights|skip_quant" logs/vis_$n.log 2>/dev/null || echo "(reused)"
    done
    echo; echo "== vision-fp16 runs (dev best, then TEST)"
    grep -hE "^done" logs/moe_seq_vis_s*.log logs/dense16_vis_s*.log
    for f in logs/vis_test_*.log; do echo "-- $f"; grep -E "^macro|^weights" $f; done
    echo; echo "== TEST table (Δ vs the fp16 model, paired bootstrap)"
    python scripts/aggregate.py --runs runs --preds test_preds.jsonl --ref fp16_512 --extra $X --out runs/results_vis
    if [ -d runs/moe_seq_s0 ]; then
        echo; echo "== same table, Δ vs 4-bit moe_seq (does an fp16 vision encoder help the adapted model?)"
        python scripts/aggregate.py --runs runs --preds test_preds.jsonl --ref moe_seq --extra $X \
            --out runs/results_vis/vs_moe_seq
    fi
} > $S 2>&1
cat $S
say "done -> $S"
