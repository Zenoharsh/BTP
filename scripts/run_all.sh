#!/usr/bin/env bash
# Whole experiment in one go, for a Kaggle "Save Version -> Save & Run All" run on 2x T4.
# Resumable: every finished step is skipped, so re-running after a crash continues where it stopped.
#
#   bash scripts/run_all.sh            # everything (~6.5 h)
#   DRY=1 bash scripts/run_all.sh      # print the plan only
#
# Results: runs/results/SUMMARY.txt (all dev lines, routing stats, merges, 3-seed table),
#          runs/results/table.md/.tex, per-run logs in logs/, bench_inputs.zip for the laptop.
cd "$(dirname "$0")/.."
mkdir -p logs runs/results
say() { echo "[$(date +%H:%M:%S)] $*"; }
run() { if [ -n "$DRY" ]; then echo "    $*"; else "$@"; fi; }

train() {   # gpu config seed
    local gpu=$1 cfg=$2 seed=$3 r=runs/$2_s$3
    if [ -f "$r/summary.json" ]; then say "skip $r (done)"; return; fi
    say "GPU$gpu train $r"
    run env CUDA_VISIBLE_DEVICES=$gpu python train.py --config configs/$cfg.yaml --seed $seed > logs/$2_s$3.log 2>&1
    grep -E "^\[dev\]|^done" logs/$2_s$3.log || say "!! $r failed, see logs/$2_s$3.log"
}

routing() { # gpu config seed: routing record + MI stats (MoE runs)
    local gpu=$1 cfg=$2 seed=$3 r=runs/$2_s$3
    [ -f "$r/best/adapters.pt" ] || [ -n "$DRY" ] || return
    if [ ! -f "$r/routing_dev.npz" ]; then
        say "GPU$gpu routing $r"
        run env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --config configs/$cfg.yaml --adapters $r/best \
            --record_routing --out $r/dev_eval.jsonl > logs/$2_s$3_eval.log 2>&1
    fi
    [ -n "$DRY" ] || python convert.py --config configs/$cfg.yaml --routing $r/routing_dev.npz --mode stats > $r/stats.txt 2>&1
}

merges() {  # gpu config seed: task-profile and full-merge accuracy (deployment variants)
    local gpu=$1 cfg=$2 seed=$3 r=runs/$2_s$3
    [ -f "$r/routing_dev.npz" ] || [ -n "$DRY" ] || return
    for mode in profile full_merge; do
        [ -f "$r/best/convert_$mode.json" ] && continue
        say "GPU$gpu $mode $r"
        run env CUDA_VISIBLE_DEVICES=$gpu python convert.py --config configs/$cfg.yaml --adapters $r/best \
            --routing $r/routing_dev.npz --mode $mode --eval > logs/$2_s$3_$mode.log 2>&1
    done
}

# ---------------------------------------------------------------- 1. base model + teacher cache
if [ ! -f runs/b0_512/dev_preds.jsonl ]; then
    say "base model eval"
    run env CUDA_VISIBLE_DEVICES=0 python evaluate.py --base --out runs/b0_512/dev_preds.jsonl > logs/b0.log 2>&1 &
fi
if [ -n "$DRY" ] || ! python scripts/cache_teacher.py --check > /dev/null 2>&1; then
    say "teacher cache"
    run env CUDA_VISIBLE_DEVICES=0 python scripts/cache_teacher.py --shard 0 --num_shards 2 > logs/cache0.log 2>&1 &
    run env CUDA_VISIBLE_DEVICES=1 python scripts/cache_teacher.py --shard 1 --num_shards 2 > logs/cache1.log 2>&1 &
fi
wait
[ -n "$DRY" ] || python scripts/cache_teacher.py --check | tail -n 3
[ -n "$DRY" ] || python scripts/cache_teacher.py --check > /dev/null 2>&1 || { say "!! teacher cache incomplete, stopping"; exit 1; }

# ---------------------------------------------------------------- 2. training, one queue per GPU
(
    train 0 moe_seq 0; routing 0 moe_seq 0; merges 0 moe_seq 0
    train 0 moe_seq 1; routing 0 moe_seq 1
    train 0 moe_seq 2; routing 0 moe_seq 2
    train 0 moe 0;     routing 0 moe 0
) 2>&1 | tee logs/gpu0.log &
(
    train 1 dense16 0
    train 1 dense16 1
    train 1 dense16 2
    train 1 moe 1;     routing 1 moe 1
    train 1 dense64 0
) 2>&1 | tee logs/gpu1.log &
wait
[ -n "$DRY" ] && exit 0

# ---------------------------------------------------------------- 3. summary
S=runs/results/SUMMARY.txt
{
    echo "== dev results per run"; grep -hE "^done" logs/*_s[0-9].log
    echo; echo "== base"; tail -n 6 logs/b0.log
    for r in runs/moe_seq_s* runs/moe_s*; do [ -f $r/stats.txt ] && { echo; echo "== routing $r"; head -n 4 $r/stats.txt; }; done
    for f in logs/*_profile.log logs/*_full_merge.log; do [ -f $f ] && { echo; echo "== $f"; grep -E "macro" $f; }; done
    echo; echo "== 3-seed table (vs dense16, paired bootstrap)"
    python scripts/aggregate.py --ref dense16 --extra base=runs/b0_512/dev_preds.jsonl
} > $S 2>&1
cat $S
zip -qr bench_inputs.zip runs/moe_seq_s0/best runs/moe_seq_s0/routing_dev.npz runs/moe_s0/best \
    runs/moe_s0/routing_dev.npz runs/dense16_s0/best 2>/dev/null
say "all done -> $S, bench_inputs.zip"
