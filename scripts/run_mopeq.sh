#!/usr/bin/env bash
# MoPEQ-style mixed precision of the frozen base (Kaggle commit, 2x T4), evaluated on the TEST split.
#   1. per-unit sensitivity (89 units: 32 vision blocks, merger, 28 x {attention, MLP}) with HQQ at
#      2/3/4/8 bits: KL to the fp16 model (both GPUs), then MoPEQ/HAWQ Hessian traces (GPU 0)
#   2. allocation at the memory of uniform 3 / 3.5 / 4-bit (exact knapsack), plus uniform baselines
#   3. base model with each plan on TEST: accuracy, weights, peak VRAM
# Attach an earlier output as an Input to reuse the fp16 / NF4 test references and any finished MoPEQ
# step (sensitivities, plans, evaluations); missing pieces are (re)computed. Resumable; results in
# runs/results_mopeq/SUMMARY.txt
cd "$(dirname "$0")/.."
mkdir -p logs runs/mopeq runs/results_mopeq
say() { echo "[$(date +%H:%M:%S)] $*"; }
T=data/v3/test.jsonl
M=runs/mopeq
pip install -q hqq
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ref in fp16_512 b0_512; do              # reuse earlier references if attached
    [ -f runs/$ref/test_preds.jsonl ] && continue
    hit=$(find /kaggle/input -path "*runs/$ref/test_preds.jsonl" 2>/dev/null | head -n 1)
    [ -n "$hit" ] && mkdir -p runs/$ref && cp "$(dirname "$hit")"/* runs/$ref/ && say "reused $ref"
done

# reuse a previous run_mopeq output (sensitivities / plans / evaluations) if attached
prev=$(find /kaggle/input -path "*runs/mopeq/sens_kl_0.json" 2>/dev/null | head -n 1)
if [ -n "$prev" ]; then
    root=$(dirname "$(dirname "$(dirname "$prev")")")
    for f in "$root"/runs/mopeq/*.json; do [ -f $M/$(basename $f) ] || cp "$f" $M/; done
    for d in "$root"/runs/hqq_*; do [ -d runs/$(basename $d) ] || cp -r "$d" runs/; done
    for f in "$root"/logs/mopeq_*.log; do [ -f logs/$(basename $f) ] || cp "$f" logs/ 2>/dev/null; done
    say "reused previous MoPEQ outputs from $root"
fi

ev() {  # gpu name extra-args...
    local gpu=$1 name=$2; shift 2
    [ -f runs/$name/test_preds.jsonl ] && return
    say "GPU$gpu eval $name"
    env CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --base --split $T --out runs/$name/test_preds.jsonl "$@" \
        > logs/mopeq_eval_$name.log 2>&1 || say "!! $name failed (logs/mopeq_eval_$name.log)"
}

# ---------------------------------------------------------------- 1. sensitivity
for s in 0 1; do
    [ -f $M/sens_kl_$s.json ] || { say "GPU$s KL sensitivity shard $s"; \
        CUDA_VISIBLE_DEVICES=$s python mopeq.py sensitivity --metric kl --calib 32 --shard $s --num_shards 2 \
            --out $M/sens_kl_$s.json > logs/mopeq_sens_kl_$s.log 2>&1 & }
done
wait
for s in 0 1; do [ -f $M/sens_kl_$s.json ] || { say "!! KL sensitivity shard $s failed"; tail -n 20 logs/mopeq_sens_kl_$s.log; exit 1; }; done

# Hessian (MoPEQ's metric): Hutchinson traces with finite-difference Hessian-vector products (exact
# double backprop does not fit a T4), split over both GPUs; uniform baselines afterwards
for s in 0 1; do
    [ -f $M/sens_hessian_$s.json ] || { say "GPU$s Hessian sensitivity shard $s"; \
        CUDA_VISIBLE_DEVICES=$s python mopeq.py sensitivity --metric hessian --hvp fd --calib 16 --n_iter 2 \
            --max_pixels 200704 --shard $s --num_shards 2 --out $M/sens_hessian_$s.json \
            > logs/mopeq_sens_hessian_$s.log 2>&1 || say "!! Hessian shard $s failed (see log); KL plans still run" & }
done
wait
HESS=""
[ -f $M/sens_hessian_0.json ] && [ -f $M/sens_hessian_1.json ] && HESS="$M/sens_hessian_0.json $M/sens_hessian_1.json"
(
    for b in 4 3; do
        [ -f $M/plan_u$b.json ] || python mopeq.py uniform --sizes $M/sens_kl_0.json $M/sens_kl_1.json --bits $b             --out $M/plan_u$b.json > /dev/null
    done
    [ -f runs/fp16_512/test_preds.jsonl ] || ev 1 fp16_512 --fp16
    [ -f runs/b0_512/test_preds.jsonl ] || ev 1 b0_512
    ev 1 hqq_u4 --mopeq_plan $M/plan_u4.json
    ev 1 hqq_u3 --mopeq_plan $M/plan_u3.json
) &
wait

# ---------------------------------------------------------------- 2. allocation
for a in 3 3.5 4; do
    python mopeq.py allocate --sens $M/sens_kl_0.json $M/sens_kl_1.json --avg_bits $a --out $M/plan_kl_$a.json \
        > logs/mopeq_alloc_kl_$a.log 2>&1
    [ -n "$HESS" ] && python mopeq.py allocate --sens $HESS --avg_bits $a \
        --out $M/plan_hess_$a.json > logs/mopeq_alloc_hess_$a.log 2>&1
done

# ---------------------------------------------------------------- 3. evaluation of the mixed plans
( for a in 3 4; do ev 0 hqq_kl_$a --mopeq_plan $M/plan_kl_$a.json; done
  [ -f $M/plan_hess_3.json ] && ev 0 hqq_hess_3 --mopeq_plan $M/plan_hess_3.json ) &
( ev 1 hqq_kl_3.5 --mopeq_plan $M/plan_kl_3.5.json
  [ -f $M/plan_hess_4.json ] && ev 1 hqq_hess_4 --mopeq_plan $M/plan_hess_4.json
  [ -f $M/plan_hess_3.5.json ] && ev 1 hqq_hess_3.5 --mopeq_plan $M/plan_hess_3.5.json ) &
wait

# ---------------------------------------------------------------- 4. summary
S=runs/results_mopeq/SUMMARY.txt
{
    echo "== plans (quantised part only)"
    for p in $M/plan_*.json; do python -c "import json,sys; d=json.load(open('$p'))['summary']; \
print(f\"{'$p'.split('/')[-1]:<22} avg {d['avg_bits']:.2f} bit  vision {d['avg_bits_vision']:.2f}  language {d['avg_bits_language']:.2f}  {d['mib']:.0f} MiB  {d['units_per_bits']}\")"; done
    echo; echo "== TEST: accuracy and GPU memory"
    for f in logs/mopeq_eval_*.log; do echo "-- $f"; grep -E "^macro|weights" $f; done
    echo; echo "== TEST table (paired bootstrap vs the fp16 model)"
    extras=""
    for d in runs/fp16_512 runs/b0_512 runs/hqq_*; do [ -f $d/test_preds.jsonl ] && extras="$extras $(basename $d)=$d/test_preds.jsonl"; done
    mkdir -p runs/_none
    python scripts/aggregate.py --runs runs/_none --preds test_preds.jsonl --ref fp16_512 --extra $extras \
        --out runs/results_mopeq
    echo; echo "== TEST table, same memory: mixed vs uniform HQQ 4-bit (paired bootstrap)"
    python scripts/aggregate.py --runs runs/_none --preds test_preds.jsonl --ref hqq_u4 --extra $extras \
        --out runs/results_mopeq/vs_u4
    echo; echo "== TEST table, same memory: mixed vs uniform HQQ 3-bit (paired bootstrap)"
    python scripts/aggregate.py --runs runs/_none --preds test_preds.jsonl --ref hqq_u3 --extra $extras \
        --out runs/results_mopeq/vs_u3
    echo; echo "== Hessian (MoPEQ) vs KL sensitivity: rank agreement and plan overlap"
    python scripts/mopeq_compare.py
    echo; echo "== most sensitive units (KL, 2-bit)"
    python - <<'EOF'
import json
s = {}
for f in ("runs/mopeq/sens_kl_0.json", "runs/mopeq/sens_kl_1.json"):
    s.update(json.load(open(f))["sens"])
for u, v in sorted(s.items(), key=lambda kv: -kv[1]["2"])[:12]:
    print(f"{u:<14} " + "  ".join(f"{b}b={x:.2e}" for b, x in v.items()))
EOF
} > $S 2>&1
cat $S
say "done -> $S"
