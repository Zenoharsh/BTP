# ANTIGRAVITY INSTRUCTIONS — BTP v3 (ICORT 2027)

> **Read this entire file before writing any code.** Work through the tasks **in order**. After each task: run its acceptance check, paste the output into your reply, and commit with the given message. **Stop and report** if any acceptance check fails. Do not continue past a failed check.

---

## 0. Context in 60 seconds

- **Project:** Qwen2-VL-2B-Instruct (4-bit NF4) + a Mixture of LoRA experts inside each of the 28 language-model MLP blocks. Base weights stay frozen; only the LoRA experts and routers train.
- **Paper:** IEEE ICORT 2027, deadline **25 Oct 2026**. The paper is about making this model **deployable on edge hardware**: 4 GB GTX 1650, CPU-only 8 GB laptop, and optionally a Raspberry Pi 5.
- **Training:** Kaggle (2× T4, 16 GB each, **no native bf16**). The developer's laptop GTX 1650 is also Turing (sm75), with the same constraints.
- **Data:** 3 domains: `document_ocr` (DocVQA), `chart_qa` (ChartQA), `spatial_reasoning` (VSR, True/False).
- **The master plan** (`BTP_Master_Plan_ICORT2027.md`) defines changes A1–A8 and P1–P9. This file is how to implement them.

### Why the old code must change (confirmed by reading the code and by tests)
1. **Router never learned.** `router.py` renormalised the top-1 weight to `p/p = 1.0`, so the router got zero task gradient (measured 1.2e-9). This is fixed in the new `layers.py`/`router.py`.
2. **"Drop rate" confusion.** `train.py` averaged counts and drop rate over 28 layers, which hid per-layer overflow. The new design is dropless by default and logs per layer.
3. **High CE on spatial samples** (CE 8–15, every one from sample IDs 200–299). Those samples are bare statements with no instruction to answer True/False. Fixed with prompt templates.
4. **Unreadable documents.** `image.thumbnail((512,512))` shrinks 1695×2025 DocVQA pages. Fixed by using the processor's `max_pixels`.
5. **`import math` missing** in `train.py` (the latest commit would crash).
6. **The model never learns to stop.** The answer mask excludes `<|im_end|>`, so the end-of-answer token is never supervised. Fixed by including it.
7. **Capacity applied only in training mode** (train/eval mismatch); a Python loop over experts; parameter count taken from packed 4-bit tensors (`numel` undercounts).

---

## 1. Rules for the agent

- `layers.py`, `router.py`, `test_v3.py` are given in the Appendix **verbatim and already tested** (14/14 pass). **Copy them exactly. Do not "improve" their logic.** If you think something in them is wrong, stop and explain.
- Never call `.to(dtype=...)` on a module that contains bitsandbytes 4-bit layers. Move only **new** modules (LoRA, router) to the device. They are created in float32 on purpose.
- Never hard-code token IDs. Get them from the tokenizer: `tok.convert_tokens_to_ids("<|image_pad|>")`, `"<|im_start|>"`, `"<|im_end|>"`.
- Every script takes a `--config` YAML and a `--seed`, and writes outputs under `runs/<run_id>/`. `run_id = f"{variant}_s{seed}_{YYYYmmdd-HHMM}"`.
- Save with every checkpoint: the config, the seed, `git rev-parse HEAD`, and `pip freeze` for torch/transformers/bitsandbytes/accelerate.
- Do not regenerate the teacher cache except in Task T6.
- One commit per task. Commit message given per task.
- Never use the test split for any choice (resolution, checkpoint, k in selective merge). Use dev.

---

## 2. Tasks

### T0 — Branch and core architecture (no GPU needed)
1. `git checkout -b v3-architecture`
2. Replace `layers.py` and `router.py` with the Appendix versions. Add `test_v3.py` from the Appendix.
3. Move legacy files into `legacy/` (do not delete; they document history): `test_smoke.py`, `sandbox.py`, `test_deepcopy.py`, `eval_and_profile_moe.py`, `benchmark.py`, `mopeq_quantize.py`, `generate_architecture.py`, `generate_presentation.py`, `BTP_Project_Presentation.pptx`, `moe_concepts_explained.md`.
4. Add `pytest` to `requirements.txt` and pin versions later in T13.

**Acceptance:** `python test_v3.py` prints `All 14 v3 tests passed.`
**Commit:** `refactor(v3): fused gate-scaled MoE-LoRA layer, dropless routing, text-only routing; archive legacy scripts`

---

### T1 — Config system
Create `configs/` with one YAML per variant and `config.py` (dataclass + YAML loader, with CLI overrides like `--set train.lr=1e-4`).

`configs/base.yaml` (shared):
```yaml
model_id: Qwen/Qwen2-VL-2B-Instruct
data:
  dir: data/v3            # built by T3
  min_pixels: 50176       # 64*28*28
  max_pixels: 401408      # 512*28*28 — final value chosen in T7 sweep
moe:
  num_experts: 4
  rank: 16
  alpha: 32
  top_k: 1
  capacity_factor: null   # dropless (A3)
  route_tokens: all       # all | text (A4)
  lb_coef: 0.01
  z_coef: 0.001
loss:
  ce: 1.0
  kd: 1.0
  kd_temperature: 2.0
train:
  epochs: 2
  batch_size: 1
  grad_accum: 8
  lr: 2.0e-4              # LoRA params
  router_lr: 1.0e-3       # router params (separate param group)
  weight_decay: 0.0
  warmup_ratio: 0.05
  scheduler: cosine
  max_grad_norm: 1.0
  log_every: 1            # optimizer steps
  eval_every_epoch: true
  gradient_checkpointing: true
```
Variant files override only what differs:
| file | overrides |
|---|---|
| `dense16.yaml` | `moe.num_experts: 1, moe.rank: 16` |
| `dense64.yaml` | `moe.num_experts: 1, moe.rank: 64, moe.alpha: 128` |
| `moe.yaml` | (nothing) |
| `moe_text.yaml` | `moe.route_tokens: text` |
| `moe_nokd.yaml` | `loss.kd: 0.0` |
| `overfit.yaml` | `train.epochs: 25`, data subset 32 samples (`data.limit: 32`), `train.grad_accum: 4` |

Note that `dense64` uses `alpha = 2*rank` like the others, so the LoRA scaling (alpha/rank = 2) is identical across variants.

**Acceptance:** `python -c "from config import load; c=load('configs/dense64.yaml'); print(c.moe.rank, c.moe.alpha, c.moe.num_experts)"` prints `64 128 1`.
**Commit:** `feat(config): YAML configs for dense16/dense64/moe/moe_text/moe_nokd/overfit`

---

### T2 — Prompts, labels, dataset class (CPU-testable)
**`prompts.py`:**
```python
TEMPLATES = {
  "document_ocr":      "{q}\nAnswer the question using a single word or phrase.",
  "chart_qa":          "{q}\nAnswer the question using a single word or phrase.",
  "spatial_reasoning": "Statement: {q}\nIs this statement true or false about the image? Answer True or False.",
}
def build_prompt(task, question): return TEMPLATES[task].format(q=question.strip())
```

**`utils.py` → `get_answer_labels`:** keep the logic, but (a) take the token IDs as arguments resolved from the tokenizer, and (b) **include `<|im_end|>`** in the labels (change `labels[i, start+3:end]` to `labels[i, start+3:end+1]`). Update the docstring. Add a unit test in `test_data.py`: for `[.., im_start, assistant, \n, 10, 11, 12, im_end, \n]` the supervised tokens are `[10, 11, 12, im_end]`.

**`data.py`** (replaces the dataset class in `train.py`):
- `VQADataset(jsonl_path, processor, limit=None)`: each line `{"uid","task","image","question","answers":[...]}`. Training target = `answers[0]`.
- **Do not resize images yourself.** Remove `thumbnail`. Resolution is controlled only by the processor (`AutoProcessor.from_pretrained(model_id, min_pixels=..., max_pixels=...)`).
- Messages: user = [image, `build_prompt(task, question)`], assistant = answer. `apply_chat_template(..., add_generation_prompt=False)` for training. For evaluation, the same user message with `add_generation_prompt=True` and no assistant turn.
- `__getitem__` returns `{"text", "image", "uid", "task"}`. The collate function returns processor outputs + `uids` + `tasks`.

**Acceptance:** `pytest test_data.py` passes (labels test + a prompt-format test).
**Commit:** `feat(data): task prompt templates, supervise im_end, processor-controlled resolution`

---

### T3 — Build v3 dataset splits (`scripts/build_dataset.py`, run on Kaggle with internet ON)
Sources (print `ds.features` first and adapt field names if they differ):
| task | HF dataset | train/dev source | test source | answer field |
|---|---|---|---|---|
| document_ocr | `pixparse/docvqa-single-page-questions` | `train` | `validation` | `answers` (list) |
| chart_qa | `HuggingFaceM4/ChartQA` | `train` | `test` (or `val`) | `label` (list) |
| spatial_reasoning | `cambridgeltl/vsr_random` | `train` | `test` | `label` 0/1 → `"False"`/`"True"`; question = `caption`; image downloaded from `image_link` (retry 3×, skip on failure) |

Rules:
- Sizes per task: **train 400, dev 50** (both from the train split, **disjoint by image**), **test 200** (from the test source). Seeded sampling (seed 0).
- De-duplicate by image hash across all splits. Assert no image appears in two splits.
- Save images as JPEG under `data/v3/images/<uid>.jpg` and write `data/v3/{train,dev,test}.jsonl`.
- Write `data/v3/manifest.json` with counts per split/task, source dataset and split names, and the sha256 of each jsonl.
- Also convert the old 300-sample `train.json` into `data/smoke.jsonl` with the same schema (uid = `smoke_<idx>`), used only for smoke runs.
- Upload `data/v3` as a Kaggle dataset (`btp-data-v3`) so later sessions do not re-download.

**Acceptance:** the script prints a table with 400/50/200 per task and `overlap check: PASS`.
**Commit:** `feat(data): deterministic 3-domain v3 splits with overlap check and manifest`

---

### T4 — Model setup (`model.py`)
```python
def build_model(cfg, device="cuda:0"):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)   # T4/1650: fp16, not bf16
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, quantization_config=bnb,
                                                            device_map={"": 0})
    tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
    for layer in get_lm_layers(model):
        dev = layer.mlp.down_proj.weight.device
        new = MoELayer(layer.mlp, tc.hidden_size, tc.intermediate_size,
                       num_experts=cfg.moe.num_experts, rank=cfg.moe.rank, alpha=cfg.moe.alpha,
                       top_k=cfg.moe.top_k, capacity_factor=cfg.moe.capacity_factor,
                       route_tokens=cfg.moe.route_tokens,
                       lb_coef=cfg.moe.lb_coef, z_coef=cfg.moe.z_coef)
        # move ONLY the new modules; never .to() the bnb base
        new.lora_gate.to(dev); new.lora_up.to(dev); new.lora_down.to(dev)
        if new.router is not None: new.router.to(dev)
        layer.mlp = new
    for n, p in model.named_parameters():
        p.requires_grad = (".lora_" in n) or (".router." in n)
    install_token_type_hook(model, tok.convert_tokens_to_ids("<|image_pad|>"))
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model
```
- **fp16 compute dtype:** T4/GTX 1650 have no native bf16. If you see NaN/inf in fp16, report it and try `bnb_4bit_compute_dtype=torch.float32` for the base. Do not silently switch.
- **Parameter report (P3):** trainable = sum of `numel` for trainable params. "Total" = **unquantized** model size: load the config and compute it, or use the HF model card value. Print trainable as a percentage of that. Never use `numel` of 4-bit packed tensors.
- `save_adapters(model, path)`: save only `lora_*` and `router.*` tensors + config + commit hash. `load_adapters(model, path)`.

**Acceptance (CPU, without loading Qwen):** a test with a tiny fake model shows that `save_adapters` then `load_adapters` reproduces outputs exactly, and that only `lora_*`/`router.*` keys are saved.
**Commit:** `feat(model): NF4 model build with MoE surgery, fp16 compute, adapter-only save/load`

---

### T5 — Real-model sanity check (`scripts/check_real_model.py`) — **Kaggle, run before anything else on GPU**
On 3 samples (one per task) from `data/smoke.jsonl`:
1. **Identity:** logits of the unmodified NF4 model vs. the surgically modified model with zero-init LoRA (B=0). Assert `max |diff| < 1e-2` (fp16 noise) and identical argmax on all answer positions.
2. **Gradient flow:** one forward + backward with CE. Assert every `lora_*.A` and every `router.gate.weight` has a non-zero, finite gradient. Note: with B=0, gradients reach B first; A and the router receive gradient only once B≠0, so run **one optimizer step** and then a second backward before asserting on A and the router.
3. **Generation:** `model.generate(max_new_tokens=16)` works with the hook installed, for both `route_tokens=all` and `text`.
4. Print: peak VRAM, number of image tokens per sample, total sequence length.

**Acceptance:** all asserts pass; paste the printout.
**Commit:** `test(real): identity, gradient-flow and generation checks on real Qwen2-VL`

---

### T6 — Teacher cache v3 (`scripts/cache_teacher.py`) — Kaggle
- Teacher = the same model, **unquantized**, fp16 (or bf16 if fp16 gives NaN; report which).
- Use the **same** `data.py` and processor settings as training (same `max_pixels`, same prompts), on `train.jsonl` and `dev.jsonl`.
- Key files by `uid`, not dataset index: `teacher_cache/v3/<uid>.pt` with `{"probs":[n,50] fp16, "indices":[n,50] int32, "mask":[L] bool, "seq_len": L, "topk_mass":[n] fp32}`. `topk_mass` is the sum of the top-50 probabilities **before** renormalisation, at temperature 2.
- The answer mask must use the new `get_answer_labels` (which includes `<|im_end|>`), shifted exactly as before (`shift_mask = valid_mask[1:]`, logits `[:-1]`).
- At train time, **assert** `seq_len == input_ids.shape[1]` for every sample. If they differ, stop: teacher and student disagree on tokenization or resolution.
- Integrity script: count files, total answer tokens, NaNs, and the `topk_mass` distribution (mean, p5).

**Acceptance:** integrity PASS for train+dev; report the `topk_mass` mean and p5.
**Commit:** `feat(kd): uid-keyed v3 teacher cache with top-k mass and length check`

---

### T7 — Training script (`train.py`, full rewrite)
CLI: `python train.py --config configs/moe.yaml --seed 0 [--set k=v ...]`
- Seeds: `random`, `numpy`, `torch` (+cuda). DataLoader with a seeded generator, `shuffle=True`.
- Loss per micro-batch:
  ```python
  out = model(**inputs)                    # do NOT pass labels
  logits = out.logits[:, :-1].float(); labels = labels[:, 1:]
  ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
  kd = sparse_kd(logits[0][cache.mask], cache.indices, cache.probs, T)  # same math as before, T^2 scaled
  aux = total_aux_loss(model)              # mean over layers, None for dense
  loss = cfg.loss.ce*ce + cfg.loss.kd*kd + (aux if aux is not None else 0)
  (loss / grad_accum).backward()
  ```
  Keep the existing sparse-KD math (renormalised teacher top-50, student full-vocab logsumexp). Cast logits to float32 before softmax/logsumexp.
- Optimizer: AdamW with two param groups (LoRA: `lr`, router: `router_lr`), cosine schedule with warmup, grad clipping `max_grad_norm`. Step every `grad_accum` micro-batches **and** at the end of each epoch.
- **Logging (A8)** to `runs/<id>/train_log.jsonl`, one line per optimizer step: step, epoch, lr, mean CE, KD, aux over the accumulation window, seconds per step, peak VRAM. Plus a routing summary for MoE: over layers → mean and **max** of `entropy_token_mean`, `overflow_rate`, `load_cv`; plus the full per-layer `post_counts` for layers 0, 13 and 27.
- Every epoch: evaluate on `dev.jsonl` with the T9 function (greedy, 32 tokens) and log per-task scores. Keep the checkpoint with the best dev average. Save adapters only.
- Print progress every 10 optimizer steps; never print per-sample walls of text.

**Acceptance (Kaggle):** `python train.py --config configs/overfit.yaml --seed 0` → train CE on the 32 samples falls below 0.1; dev evaluation runs; for the MoE config, `entropy_token_mean` falls below 1.35 in at least some layers by the end (the router is learning). Paste the last 5 log lines.
**Commit:** `feat(train): v3 training loop with param groups, cosine schedule, per-layer routing telemetry`

---

### T8 — Resolution sweep for the base model (B0) — Kaggle
Run `evaluate.py` (T9) for the **untrained** base model on `dev.jsonl` at `max_pixels ∈ {256, 512, 768} × 28 × 28`. Record score per task, mean visual tokens, latency. Pick the **smallest** setting within 1 point of the best average dev score. Write it into `configs/base.yaml`. This value is then fixed for everything, including regenerating the teacher cache in T6 if it changed.

**Order note:** run T8 before T6, or rerun T6 after T8 if `max_pixels` changed.
**Commit:** `exp(b0): resolution sweep, fix max_pixels=<value>`

---

### T9 — Evaluation (`evaluate.py`)
- `evaluate(model, processor, jsonl, out_path, max_new_tokens=32)`: greedy (`do_sample=False`), batch size 1. Saves `preds.jsonl` (uid, task, pred, answers, score, latency).
- Metrics (`metrics.py`, unit-tested):
  - `document_ocr`: **ANLS**, τ=0.5, max over ground truths. Strip whitespace, lowercase, collapse spaces in both strings; if the pred is empty the score is 0.
  - `chart_qa`: **relaxed accuracy**. If both parse as numbers (strip `%`, `,`), correct when `|p - g| / |g| ≤ 0.05` (exact when g = 0); otherwise case-insensitive exact match.
  - `spatial_reasoning`: accuracy after mapping the pred to True/False (accept `true/yes` → True, `false/no` → False; anything else is wrong).
  - Also exact match for all tasks.
- Report per task and macro average. Also an option `--record_routing` that sets `layer.record=True` and saves per-sample, per-layer expert ID histograms over **text tokens** to `routing_dev.npz` (used for MI and conversions).

**Acceptance:** `pytest test_metrics.py` covers ANLS (identical → 1, totally different → 0, threshold case), relaxed accuracy (5% boundary, percentages), and True/False mapping.
**Commit:** `feat(eval): ANLS / relaxed-acc / TF accuracy evaluation with routing recording`

---

### T10 — Deployment converters (`convert.py`)
Inputs: a trained MoE checkpoint + `routing_dev.npz`.
1. **Statistics:** per layer, the expert frequency `f[l, e]`; per task, `f_task[t, l, e]`; per layer, **MI(task; expert)** in bits from the joint counts.
2. **Full merge:** for every layer, `layer.mode = "fixed"`, `layer.fixed_g = f[l]` (frequencies sum to 1).
3. **Selective merge (k):** keep `mode="routed"` for the k layers with the highest MI; set fixed mode with `f[l]` for the rest. k ∈ {0, 4, 8, 14, 28} (k=0 is the full merge, k=28 is the original model). Choose the reported k on **dev**.
4. **Profile adapters:** for task t, in every layer, use `fixed_g = onehot(argmax_e f_task[t, l, e]) * gbar[l, e]`, where `gbar` is the mean gate value observed for that expert (record it in T9). At evaluation, the task label selects the profile (in deployment the application knows its task: a document reader, a chart reader, a scene checker).
5. **Export (for llama.cpp / Pi):** load the **unquantized fp16** base on CPU; for every layer in fixed mode, add `layer.merged_delta_weights(fixed_g)` to `mlp.{gate,up,down}_proj.weight`; `save_pretrained` the result **as a standard Qwen2-VL checkpoint** (no MoE code needed to load it). The same applies to dense16/dense64 (g = [1]). One export per profile.
6. **Unit test:** on a tiny model, the exported dense weights reproduce fixed-mode outputs (the Appendix test `test_fixed_mode_equals_merged_weights` already proves the math).

**Acceptance:** `python convert.py --ckpt runs/<moe_run> --mode full_merge --eval dev` runs and prints dev scores; the exported checkpoint loads with plain `Qwen2VLForConditionalGeneration.from_pretrained` and generates.
**Commit:** `feat(convert): full/selective merge, profile adapters, standard-checkpoint export`

---

### T11 — Edge benchmark (`benchmark.py`, full rewrite) — run on the GTX 1650 laptop and CPU
- Loads **trained** artifacts only: the base model, adapter checkpoints, or exported merged checkpoints. Never a freshly constructed model.
- Protocol: batch 1, fixed `max_pixels` (from T8), 5 warm-up + 30 timed queries (10 per task, fixed uids from test), `torch.cuda.synchronize()` around timings.
- Metrics per configuration: time-to-first-token (prefill), decode tokens/s, end-to-end latency per answer, peak VRAM (`max_memory_allocated`) or peak RSS on CPU (`psutil`), model load time, file size on disk. Report the median and IQR.
- Configurations: `b0`, `dense16` (unmerged and merged), `dense64`, `moe` (routed), `moe_text`, `moe→full_merge`, `moe→selective(k*)`, `moe→profiles`.
- Output: `bench/<device>.csv` + the exact environment (GPU name, driver, torch/cuda versions).
- CPU mode: `--device cpu`, fp32 base (bitsandbytes 4-bit is GPU-only); fewer queries are acceptable (10).
- Stretch (only if time): llama.cpp (`convert_hf_to_gguf.py` on the exported checkpoint + `--mmproj`, Q4_K_M), run with `llama-mtmd-cli` on the laptop CPU and the Pi 5. **De-risk early:** in week 1, try the conversion on the **base** Qwen2-VL-2B first.

**Acceptance:** `python benchmark.py --device cuda --configs b0` runs on the GTX 1650 and writes a CSV.
**Commit:** `feat(bench): trained-artifact edge benchmark (GPU/CPU), TTFT/decode/e2e/peak memory`

---

### T12 — Analysis and figure scripts (`analysis/`)
- `tables.py`: main table (per-task score, macro average, mean ± std over seeds) for b0/dense16/dense64/moe/moe_text/moe_nokd; edge table from the benchmark CSVs.
- `figures.py`: (1) accuracy vs latency Pareto across all configurations and conversions; (2) heatmap of MI(task; expert) per layer; (3) training curves (CE, KD, routing entropy); (4) p(expert | task) for 3 representative layers.
- All figures as vector PDF, IEEE column width (3.5 in), font size ≥ 8 pt.

**Commit:** `feat(analysis): paper tables and figures`

---

### T13 — Reproducibility
`requirements.txt` with pinned versions from the Kaggle environment; `README.md` with the exact run order; scripts write the environment and commit hash. **Anonymize for review:** remove the author name/username from code comments, README and figures; the paper links to an anonymized mirror (anonymous.4open.science), not the GitHub repo.
**Commit:** `chore: pinned env, run-book, anonymization`

---

## 3. Kaggle run-book (order of GPU work)

| # | Command | GPU-h (est.) |
|---|---|---|
| 1 | `python scripts/build_dataset.py` (T3, internet on; also creates `data/smoke.jsonl`) | 0.5 (mostly CPU/network) |
| 2 | `python scripts/check_real_model.py` (T5) | 0.3 |
| 3 | `python evaluate.py --b0 --sweep max_pixels` (T8) | 1.5 |
| 4 | `python scripts/cache_teacher.py --splits train dev` (T6) | 1.5 |
| 5 | `python train.py --config configs/overfit.yaml --seed 0` (T7 gate) | 0.5 |
| 6 | **Pair A** — `CUDA_VISIBLE_DEVICES=0 python train.py --config configs/dense16.yaml --seed 0 &` and `CUDA_VISIBLE_DEVICES=1 python train.py --config configs/dense64.yaml --seed 0 &` | ~4 (wall) |
| 7 | **Pair B** — `moe s0` ‖ `moe_text s0` | ~4 |
| 8 | **Pair C** — `moe_nokd s0` ‖ `dense16 s1` | ~4 |
| 9 | **Pair D** — `dense64 s1` ‖ `moe s1` | ~4 |
| 10 | `evaluate.py` on test for all checkpoints + `--record_routing` on dev for MoE runs | 2 |
| 11 | `convert.py` all modes on `moe s0` (+ `moe_text s0` if it is the main model) | 1.5 |

Use `nohup ... > runs/<id>.out 2>&1 &` so a dropped notebook connection doesn't kill runs. Save `runs/` to a Kaggle dataset or Drive after each pair.

**Kill switches:** if the T7 overfit gate fails, stop all training and report. If time runs short, drop `moe_nokd`, then the seed-1 runs.

---

## 4. Do NOT
- Do not modify `layers.py` / `router.py` / `test_v3.py` logic.
- Do not cast bnb modules with `.to(dtype)`.
- Do not use bf16 anywhere on T4/GTX 1650 without reporting it.
- Do not evaluate or select on the test split before the final evaluation step.
- Do not benchmark a freshly built model as if it were trained.
- Do not print per-sample tensors in loops (it slows Kaggle and floods logs).
- Do not change hyperparameters between variants other than those listed in T1.

---

## Appendix — verified source files (copy verbatim)

### `router.py`
```python
"""
Top-k router for the shared-base Mixture-of-LoRA-Experts (v3).

Design notes (see BTP_Master_Plan_ICORT2027.md, change A1/A5):
- The router is always computed in float32 (T4 / GTX 1650 have no native bf16).
- It returns the FULL softmax probabilities. The MoE layer turns them into
  per-expert gate values g = (E / k) * p_e for the selected experts.
  We deliberately do NOT renormalise the top-k weights: for k=1 that makes
  the gate identically 1.0 and the router receives zero task gradient
  (measured 1.2e-9 on the old code).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=1, init_std=1e-4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=torch.float32)
        nn.init.normal_(self.gate.weight, std=init_std)

    def forward(self, x):
        """x: [N, H] (any dtype). Returns logits [N,E] fp32, probs [N,E] fp32, top-k indices [N,k]."""
        logits = self.gate(x.to(self.gate.weight.dtype))
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_idx = torch.topk(probs, self.top_k, dim=-1).indices
        return logits, probs, top_idx
```

### `layers.py`
```python
"""
Shared-base Mixture-of-LoRA-Experts layer (v3).

Each Qwen2 MLP block  y = down( act(gate(x)) * up(x) )  becomes

    gate'(x) = gate(x) + sum_e g_e(x) * dGate_e(x)
    up'(x)   = up(x)   + sum_e g_e(x) * dUp_e(x)
    y        = down(h) + sum_e g_e(x) * dDown_e(h),   h = act(gate'(x)) * up'(x)

where dX_e = scaling * B_e A_e is expert e's LoRA delta and g(x) is the gate vector:
    routed mode : g_e = (E/k) * p_e(x) for the k selected experts, 0 otherwise      (A1)
    fixed  mode : g   = a constant vector (used for merging / profile adapters)      (A7)
    dense  mode : E == 1, g = 1 (plain LoRA baseline with identical code path)       (A6)

Properties (all tested in test_v3.py):
- Base MLP is computed ONCE for every token and is never scaled  -> exact identity when B = 0.
- No Python loop over experts: the E low-rank adapters of a projection are stacked into one
  [E*r, in] A matrix and one [out, E*r] B matrix; the gate is applied as a block mask on the
  rank dimension (two GEMMs per projection, same FLOPs as one rank-E*r LoRA).            (A2)
- Dropless by default; optional capacity is a mask on g (overflow -> base only). Training and
  eval behave identically.                                                              (A3)
- route_tokens="text": tokens flagged as image tokens get g = 0 (base only).            (A4)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from router import TopKRouter


class StackedLoRA(nn.Module):
    """E independent LoRA adapters for one linear projection, stored stacked."""

    def __init__(self, in_features, out_features, num_experts, rank, alpha):
        super().__init__()
        self.num_experts, self.rank = num_experts, rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(num_experts * rank, in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, num_experts * rank, dtype=torch.float32))
        for e in range(num_experts):  # same init as nn.Linear / PEFT LoRA-A, per expert block
            nn.init.kaiming_uniform_(self.A.data[e * rank:(e + 1) * rank], a=math.sqrt(5))

    def forward(self, x, g):
        """x: [N, in]; g: [N, E] gate values. Returns delta [N, out] in x.dtype."""
        z = F.linear(x.to(self.A.dtype), self.A)                       # [N, E*r]
        z = z * g.to(z.dtype).repeat_interleave(self.rank, dim=-1)      # block mask / gate
        return (F.linear(z, self.B) * self.scaling).to(x.dtype)

    def expert_delta_weight(self, e):
        """Dense weight delta of expert e: scaling * B_e @ A_e  -> [out, in]."""
        sl = slice(e * self.rank, (e + 1) * self.rank)
        return self.scaling * (self.B[:, sl] @ self.A[sl])


class MoELayer(nn.Module):
    def __init__(self, base_mlp, hidden_size, intermediate_size, num_experts=4, rank=16, alpha=32,
                 top_k=1, capacity_factor=None, route_tokens="all",
                 lb_coef=0.01, z_coef=1e-3):
        super().__init__()
        assert route_tokens in ("all", "text")
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False
        self.act_fn = getattr(base_mlp, "act_fn", nn.SiLU())
        self.num_experts, self.top_k = num_experts, top_k
        self.capacity_factor = capacity_factor
        self.route_tokens = route_tokens
        self.lb_coef, self.z_coef = lb_coef, z_coef

        self.router = TopKRouter(hidden_size, num_experts, top_k) if num_experts > 1 else None
        self.lora_gate = StackedLoRA(hidden_size, intermediate_size, num_experts, rank, alpha)
        self.lora_up = StackedLoRA(hidden_size, intermediate_size, num_experts, rank, alpha)
        self.lora_down = StackedLoRA(intermediate_size, hidden_size, num_experts, rank, alpha)

        # runtime state
        self.mode = "routed"          # "routed" | "fixed"
        self.fixed_g = None           # [E] tensor used in fixed mode
        self.token_is_image = None    # [N] bool, set by the model-level pre-hook (A4)
        self.record = False           # if True, keep per-token expert ids for MI analysis
        self.aux_loss = None          # differentiable LB + z loss of the last forward
        self.metrics = {}             # detached telemetry of the last forward (A8)
        self.last_expert_ids = None

    # ------------------------------------------------------------------ gates
    def _image_mask(self, n, device):
        m = self.token_is_image
        if m is None or m.numel() != n:      # e.g. decode steps: only new text tokens
            return torch.zeros(n, dtype=torch.bool, device=device)
        return m.to(device)

    def _routed_gates(self, x):
        n, E, k = x.shape[0], self.num_experts, self.top_k
        logits, probs, top_idx = self.router(x)
        sel = torch.zeros_like(probs).scatter(-1, top_idx, 1.0)          # [N,E] one-hot (k-hot)
        routable = torch.ones(n, dtype=torch.bool, device=x.device)
        if self.route_tokens == "text":
            routable = ~self._image_mask(n, x.device)
        sel = sel * routable.unsqueeze(-1)
        pre_counts = sel.sum(0)

        if self.capacity_factor is not None:                              # optional, A3
            n_r = int(routable.sum().item())
            cap = int(math.ceil(n_r * k / E * self.capacity_factor))
            # keep the highest-probability assignments per expert
            score = torch.where(sel > 0, probs, torch.full_like(probs, -1.0))
            rank_in_expert = score.argsort(0, descending=True).argsort(0)
            sel = sel * (rank_in_expert < cap)
        else:
            cap = None

        g = sel * probs * (E / k)                                          # A1
        # ---- auxiliary losses on routable tokens only, per layer
        if routable.any():
            p_r, s_r = probs[routable], (sel[routable] > 0).float()
            f = s_r.mean(0) / k                     # fraction of assignments per expert
            P = p_r.mean(0)
            lb = E * torch.sum(f * P)
            z = torch.logsumexp(logits[routable], -1).pow(2).mean()
            self.aux_loss = self.lb_coef * lb + self.z_coef * z
        else:
            self.aux_loss = logits.sum() * 0.0

        with torch.no_grad():
            post_counts = (sel > 0).float().sum(0)
            n_r = int(routable.sum().item())
            pr = probs[routable] if n_r > 0 else probs
            tok_ent = -(pr * pr.clamp_min(1e-9).log()).sum(-1).mean()
            pm = pr.mean(0)
            self.metrics = {
                "tokens": n, "routable_tokens": n_r, "capacity": cap,
                "pre_counts": pre_counts.tolist(), "post_counts": post_counts.tolist(),
                "overflow": float(pre_counts.sum() - post_counts.sum()),
                "overflow_rate": float((pre_counts.sum() - post_counts.sum()) / max(n_r * k, 1)),
                "entropy_token_mean": float(tok_ent),
                "entropy_of_mean": float(-(pm * pm.clamp_min(1e-9).log()).sum()),
                "mean_prob": pm.tolist(),
                "top1_conf": float(pr.max(-1).values.mean()),
                "load_cv": float(post_counts.std() / post_counts.mean().clamp_min(1e-9)),
            }
            if self.record:
                ids = top_idx[:, 0].clone()
                ids[~routable] = -1
                self.last_expert_ids = ids.cpu()
        return g

    def _gates(self, x):
        n, E = x.shape[0], self.num_experts
        if E == 1:                                                        # dense LoRA baseline
            self.aux_loss, self.metrics = None, {}
            return torch.ones(n, 1, device=x.device)
        if self.mode == "fixed":                                          # merged / profile
            self.aux_loss, self.metrics = None, {}
            g = self.fixed_g.to(x.device, torch.float32).expand(n, E).clone()
            if self.route_tokens == "text":
                g[self._image_mask(n, x.device)] = 0.0
            return g
        return self._routed_gates(x)

    # ---------------------------------------------------------------- forward
    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        g = self._gates(x)
        b = self.base_mlp
        gate = b.gate_proj(x) + self.lora_gate(x, g)
        up = b.up_proj(x) + self.lora_up(x, g)
        h = self.act_fn(gate) * up
        y = b.down_proj(h) + self.lora_down(h, g)
        return y.reshape(shape)

    # ------------------------------------------------------- deployment (A7)
    def merged_delta_weights(self, g):
        """Dense weight deltas equal to fixed-mode behaviour with gate vector g (length E)."""
        out = {}
        for name, mod in (("gate_proj", self.lora_gate), ("up_proj", self.lora_up),
                          ("down_proj", self.lora_down)):
            out[name] = sum(float(g[e]) * mod.expert_delta_weight(e) for e in range(self.num_experts))
        return out


# ---------------------------------------------------------------- model helpers
def get_lm_layers(model):
    """Consistent Qwen2-VL language-layer traversal (handles both HF layouts)."""
    m = model.model
    return m.language_model.layers if hasattr(m, "language_model") else m.layers


def moe_layers(model):
    return [l.mlp for l in get_lm_layers(model) if isinstance(l.mlp, MoELayer)]


def install_token_type_hook(model, image_token_id):
    """Sets token_is_image on every MoE layer from input_ids before each forward.
    Works for training and for generate(): during decoding input_ids holds only the
    new (text) tokens, so the mask is all-False."""
    layers = moe_layers(model)

    def hook(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = None if ids is None else (ids == image_token_id).reshape(-1)
        for l in layers:
            l.token_is_image = mask
    return model.register_forward_pre_hook(hook, with_kwargs=True)


def total_aux_loss(model):
    """Mean (not sum) of per-layer router losses (A5)."""
    losses = [l.aux_loss for l in moe_layers(model) if l.aux_loss is not None]
    return torch.stack(losses).mean() if losses else None
```

### `test_v3.py`
```python
"""CPU unit tests for the v3 MoE-LoRA layer. Run:  python test_v3.py   (or pytest test_v3.py)"""
import math
import torch
import torch.nn as nn
from layers import MoELayer, StackedLoRA

H, I = 32, 64
torch.manual_seed(0)


class DummyMLP(nn.Module):  # mirrors Qwen2MLP
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def make(E=4, **kw):
    base = DummyMLP()
    return base, MoELayer(base, H, I, num_experts=E, rank=4, alpha=8, **kw)


def randomize_B(m, std=0.1):
    for mod in (m.lora_gate, m.lora_up, m.lora_down):
        nn.init.normal_(mod.B, std=std)


def manual_reference(m, x, g):
    """Explicit per-expert loop implementation used to check the fused kernel."""
    b = m.base_mlp
    def d(mod, inp):
        return sum(g[:, e:e + 1] * (inp @ mod.expert_delta_weight(e).T) for e in range(m.num_experts))
    gate = b.gate_proj(x) + d(m.lora_gate, x)
    up = b.up_proj(x) + d(m.lora_up, x)
    h = m.act_fn(gate) * up
    return b.down_proj(h) + d(m.lora_down, h)


def test_identity_at_init():
    base, m = make()
    x = torch.randn(2, 10, H)
    assert torch.allclose(m(x), base(x), atol=1e-6)


def test_router_gets_task_gradient():
    """The bug in the old code: router grad ~1e-9. Must now be clearly non-zero."""
    _, m = make()
    randomize_B(m)
    x = torch.randn(1, 50, H)
    m(x).pow(2).mean().backward()          # task loss only, no aux loss
    assert m.router.gate.weight.grad.abs().max().item() > 1e-4


def test_fused_equals_loop():
    _, m = make()
    randomize_B(m)
    x = torch.randn(40, H)
    with torch.no_grad():
        y = m(x)
        g = m._gates(x)
        ref = manual_reference(m, x, g)
    assert torch.allclose(y, ref, atol=1e-5)


def test_gate_value_top1():
    _, m = make()
    x = torch.randn(30, H)
    g = m._gates(x)
    assert ((g > 0).sum(-1) == 1).all()                # exactly one expert per token
    # with near-uniform init, the selected gate is ~ E * 0.25 = 1
    assert torch.allclose(g.sum(-1), torch.ones(30), atol=1e-2)


def test_train_eval_identical():
    _, m = make()
    randomize_B(m)
    x = torch.randn(25, H)
    m.train(); a = m(x)
    m.eval(); b = m(x)
    assert torch.equal(a, b)


def test_text_only_routing():
    base, m = make(route_tokens="text")
    randomize_B(m)
    x = torch.randn(20, H)
    img = torch.zeros(20, dtype=torch.bool); img[:12] = True
    m.token_is_image = img
    y = m(x)
    assert torch.allclose(y[img], base(x[img]), atol=1e-6)            # image tokens: base only
    assert not torch.allclose(y[~img], base(x[~img]), atol=1e-4)      # text tokens: adapted
    assert m.metrics["routable_tokens"] == 8
    m.token_is_image = None                                           # decode step: all text
    assert m.metrics is not None and m(x[:1]).shape == (1, H)


def test_capacity_accounting():
    _, m = make(capacity_factor=1.25)
    with torch.no_grad():                    # force every token to expert 0
        m.router.gate.weight.zero_(); m.router.gate.weight[0, 0] = 50.0
    x = torch.randn(100, H); x[:, 0] = 1.0
    g = m._gates(x)
    cap = math.ceil(100 / 4 * 1.25)          # 32
    mt = m.metrics
    assert mt["capacity"] == cap
    assert mt["pre_counts"] == [100.0, 0.0, 0.0, 0.0]
    assert mt["post_counts"] == [float(cap), 0.0, 0.0, 0.0]
    assert mt["overflow"] == 100 - cap
    assert abs(mt["overflow_rate"] - (100 - cap) / 100) < 1e-6
    assert int((g.sum(-1) == 0).sum()) == 100 - cap      # overflow tokens -> base only
    assert sum(mt["pre_counts"]) == mt["routable_tokens"]


def test_layer_average_hides_overflow():
    """Reproduces the old log pattern: averaged counts <= capacity, yet overflow > 0."""
    counts = torch.tensor([[120., 40, 36, 30], [10, 90, 60, 66]])
    cap = math.ceil(226 * 1.25 / 4)
    overflow = (counts - cap).clamp_min(0).sum(1) / 226
    assert counts.mean(0).max() <= cap and overflow.mean() > 0


def test_fixed_mode_equals_merged_weights():
    base, m = make()
    randomize_B(m)
    w = torch.tensor([0.1, 0.4, 0.3, 0.2])
    m.mode, m.fixed_g = "fixed", w
    x = torch.randn(15, H)
    with torch.no_grad():
        y = m(x)
        dW = m.merged_delta_weights(w)
        merged = DummyMLP()
        merged.load_state_dict(base.state_dict())
        for k_, d in dW.items():
            getattr(merged, k_).weight += d
        assert torch.allclose(y, merged(x), atol=1e-5)


def test_dense_lora_baseline():
    base, m = make(E=1)
    randomize_B(m)
    x = torch.randn(9, H)
    with torch.no_grad():
        dW = m.merged_delta_weights([1.0])
        merged = DummyMLP(); merged.load_state_dict(base.state_dict())
        for k_, d in dW.items():
            getattr(merged, k_).weight += d
        assert torch.allclose(m(x), merged(x), atol=1e-5)
    assert m.router is None


def test_aux_loss_trains_router():
    _, m = make()
    x = torch.randn(64, H)
    m(x)
    m.aux_loss.backward()
    assert m.router.gate.weight.grad is not None


def test_only_lora_and_router_trainable():
    _, m = make()
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.startswith(("lora_", "router.")) for n in names)
    assert not any(n.startswith("base_mlp") for n in names)



def test_bf16_activations():
    base, m = make()
    base.to(torch.bfloat16)                    # base in bf16 like the NF4 compute dtype
    randomize_B(m)
    x = torch.randn(1, 12, H, dtype=torch.bfloat16)
    y = m(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    y.float().pow(2).mean().backward()
    assert m.lora_gate.A.grad is not None and m.router.gate.weight.grad.abs().max() > 0


def test_token_type_hook():
    from layers import install_token_type_hook

    class Layer(nn.Module):
        def __init__(self, mlp): super().__init__(); self.mlp = mlp

    class Inner(nn.Module):
        def __init__(self, layers): super().__init__(); self.layers = nn.ModuleList(layers)

    class Model(nn.Module):
        def __init__(self, layer):
            super().__init__(); self.model = Inner([layer]); self.emb = nn.Embedding(100, H)
        def forward(self, input_ids=None):
            return self.model.layers[0].mlp(self.emb(input_ids))

    base, m = make(route_tokens="text")
    randomize_B(m)
    model = Model(Layer(m))
    install_token_type_hook(model, image_token_id=7)
    ids = torch.tensor([[7, 7, 7, 3, 4]])
    y = model(input_ids=ids)
    emb = model.emb(ids)
    assert torch.allclose(y[0, :3], base(emb[0, :3]), atol=1e-6)    # image tokens -> base
    y2 = model(input_ids=torch.tensor([[5]]))                         # decode step
    assert y2.shape == (1, 1, H) and m.metrics["routable_tokens"] == 1

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} v3 tests passed.")
```
