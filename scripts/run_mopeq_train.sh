#!/usr/bin/env bash
# MoE-LoRA training on a MoPEQ-style mixed-precision base (Kaggle commit, 2x T4), TEST split.
#   moe_seq_mopeq35 : moe_seq on the KL-guided plan at the memory of uniform 3.5-bit (3 seeds, GPU 0)
#   moe_seq_hqq4    : control, moe_seq on uniform HQQ 4-bit (3 seeds, GPU 1)
# Compared with the fp16 base model and with moe_seq / dense16 on bitsandbytes NF4.
# Attach as Inputs: the run_mopeq output (plans) and an output holding teacher_cache/v3 and the NF4 test
# predictions (e.g. the run_vision version); anything missing is recomputed (the teacher cache takes
# ~30 min on both GPUs). Resumable; results in runs/results_mopeq_train/SUMMARY.txt
cd "$(dirname "$0")/.."
mkdir -p logs runs/mopeq runs/results_mopeq_train
say() { echo "[$(date +%H:%M:%S)] $*"; }
T=data/v3/test.jsonl
pip install -q hqq
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() >= 2 else 1)" || {
    say "!! needs 2 GPUs (Settings -> Accelerator -> GPU T4 x2); found: $(python -c 'import torch; print(torch.cuda.device_count())')"
    exit 1; }

fetch() {  # path-suffix -> copies the first match under /kaggle/input to ./path-suffix
    [ -e "$1" ] && return
    local hit; hit=$(find /kaggle/input -path "*/$1" 2>/dev/null | head -n 1)
    [ -n "$hit" ] && { mkdir -p "$(dirname "$1")"; cp -r "$hit" "$1"; say "reused $1"; }
}
fetch teacher_cache/v3
for p in plan_kl_3.5 plan_u4 plan_kl_4 plan_kl_3 plan_u3; do fetch runs/mopeq/$p.json; done
for n in fp16_512 b0_512 hqq_kl_3.5 hqq_u4; do fetch runs/$n/test_preds.jsonl; done
for r in moe_seq_s0 moe_seq_s1 moe_seq_s2 dense16_s0 dense16_s1 dense16_s2; do fetch runs/$r/test_preds.jsonl; done

for p in plan_kl_3.5 plan_u4; do
    [ -f runs/mopeq/$p.json ] || { say "!! runs/mopeq/$p.json not found: attach the run_mopeq output"; exit 1; }
done

# teacher cache (fp16 teacher, 512 px; independent of how the student base is quantised)
if ! python scripts/cache_teacher.py --check > /dev/null 2>&1; then
    say "teacher cache missing or incomplete: building it on both GPUs"
    for s in 0 1; do
        CUDA_VISIBLE_DEVICES=$s python scripts/cache_teacher.py --shard $s --num_shards 2 > logs/cache_$s.log 2>&1 &
    done
    wait
    python scripts/cache_teacher.py --check > /dev/null 2>&1 || {
        say "!! teacher cache incomplete"; tail -n 15 logs/cache_0.log logs/cache_1.log; exit 1; }
fi

train_eval() { # gpu config seed
    local gpu=$1 cfg=$2 seed=$3 r=runs/$2_s$3
    if [ ! -f $r/summary.json ]; then
        say "GPU$gpu train $r"
        env CUDA_VISIBLE_DEVICES=$gpu python train.py --config configs/$cfg.yaml --seed $seed > logs/${cfg}_s${seed}.log 2>&1
        grep -E "^done" logs/${cfg}_s${seed}.log || say "!! $r failed (logs/${cfg}_s${seed}.log)"
    fi
    [ -f $r/best/adapters.pt ] && [ ! -f $r/test_preds.jsonl ] && env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py \
        --config configs/$cfg.yaml --adapters $r/best --split $T --out $r/test_preds.jsonl > logs/mt_test_${cfg}_s${seed}.log 2>&1
}

( for s in 0 1 2; do train_eval 0 moe_seq_mopeq35 $s; done ) &
( for s in 0 1 2; do train_eval 1 moe_seq_hqq4 $s; done ) &
wait

S=runs/results_mopeq_train/SUMMARY.txt
{
    echo "== dev best per run"
    grep -hE "^done" logs/moe_seq_mopeq35_s*.log logs/moe_seq_hqq4_s*.log
    echo; echo "== TEST per run (score / image tokens / latency, then memory)"
    for f in logs/mt_test_*.log; do echo "-- $f"; grep -E "^macro|weights" $f; done
    extras=""
    for n in fp16_512 b0_512 hqq_kl_3.5 hqq_u4; do
        [ -f runs/$n/test_preds.jsonl ] && extras="$extras $n=runs/$n/test_preds.jsonl"
    done
    echo; echo "== TEST table vs the fp16 base model (paired bootstrap)"
    python scripts/aggregate.py --runs runs --preds test_preds.jsonl --ref fp16_512 --extra $extras \
        --out runs/results_mopeq_train
    echo; echo "== TEST table vs moe_seq on bitsandbytes NF4 (paired bootstrap)"
    python scripts/aggregate.py --runs runs --preds test_preds.jsonl --ref moe_seq --extra $extras \
        --out runs/results_mopeq_train/vs_nf4
} > $S 2>&1
cat $S
say "done -> $S"
