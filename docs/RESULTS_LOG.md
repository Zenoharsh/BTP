# Results log

Every number reported so far, copied from the run outputs (Kaggle 2x T4 unless noted), so nothing
depends on chat history. Scores are macro over chart_qa / document_ocr / spatial_reasoning (ANLS for
documents, relaxed accuracy for charts, accuracy for VSR). "±" = std over seeds. Δ / CI / p = paired
bootstrap from `scripts/aggregate.py` (p = P(diff <= 0), one-sided).

Setup: Qwen2-VL-2B-Instruct, base frozen in 4-bit NF4 (bitsandbytes, double quant, fp16 compute),
MoE-LoRA (4 experts, rank 16, top-1) in all 28 language MLPs, KD from the fp16 model (T=2, top-50 +
tail bucket), 2 epochs, best epoch on dev. 1200 / 150 / 600 train / dev / test samples (400/50/200 per
task). Default resolution max_pixels = 401408 ("512 px").

## Base model resolution sweep (dev, NF4)
| max_pixels | macro | chart | doc | spatial | img tokens | latency T4 |
|---|---|---|---|---|---|---|
| 200704 (256) | 70.46 | 70.0 | 75.4 | 66.0 | 238.5 | 0.40 s |
| 401408 (512) | 75.92 | 68.0 | 87.8 | 72.0 | 429.4 | 0.57-0.60 s |
| 602112 (768) | 76.50 | 62.0 | 95.5 | 72.0 | 549.1 | 0.70-0.74 s |

## Teacher cache (T6)
1350 files (train + dev), 5924 answer tokens. Teacher top-1 = gold answer token on 87.7% / 88.9% (two
shards). top-50 mass at T=2: mean 0.438, p5 0.157, min 0.076 -> motivated the tail-bucket KD loss.

## Overfit smoke test (32 samples, 25 epochs)
CE 0.56 -> 0.01-0.05; KD 0.02-0.17 (real KL after the tail-bucket fix); router entropy 1.386 -> 1.30.

## Dev results (150 samples), best epoch
| run | s0 | s1 | s2 |
|---|---|---|---|
| dense16 | 84.82 (ep1) | 84.76 (ep2) | 84.10 (ep2) |
| dense64 | 82.42 (ep2) | – | – |
| moe (token routing) | 83.07 (ep1) | 84.63 (ep2) | – |
| moe_seq (sequence routing) | 84.55 (ep2) | 84.71 (ep2) | 85.27 (ep2) |

Aggregate (vs dense16): base 75.9 (−8.6) · dense16 84.6 ± 0.4 · dense64 82.4 (−2.1 [−5.0, +0.2], p 0.959)
· moe 83.9 ± 1.1 (−0.7 [−2.7, +1.2], p 0.757) · moe_seq 84.8 ± 0.4 (+0.3 [−0.9, +1.6], p 0.341).
Per task dev: dense16 74.7 / 89.7 / 89.3; moe_seq 74.0 / 88.5 / 92.0.
Re-running the same seeds reproduced every dev score exactly.

## Routing specialisation (dev, text tokens of the prompt), MI(task; expert) in bits (max 1.585)
| run | mean MI | max MI (layer) | mean expert entropy (nats, max 1.386) |
|---|---|---|---|
| moe_s0 | 0.051 | 0.129 (L22) | 1.067 |
| moe_s1 | 0.046 | 0.088 (L24) | 1.077 |
| moe_seq_s0 | 0.526 | 1.263 (L18) | 0.585 |
| moe_seq_s1 | 0.469 | 1.160 (L7) | 0.558 |
| moe_seq_s2 | 0.436 | 0.985 (L9) | 0.575 |

moe_seq_s0 L18: chart -> e1 (1.00), document -> e3 (0.84), spatial -> e2 (0.94). Token routing at its
most specific layers splits chart+document vs spatial only.

## Deployment merges (convert.py)
- moe_s0 dev: full merge 81.82 · selective k=4 81.82, k=8 83.49, k=14 82.49, k=28 (routed) 83.07 · task
  profiles 83.82.
- moe_seq_s0 dev: task profiles 85.22 · full merge 83.89 (routed 84.55).
- moe_seq task profiles on TEST: 76.93 / 78.15 / 77.52 (s0/s1/s2) vs routed 76.94 / 78.66 / 77.59.

## Held-out TEST (600 samples), 512 px
| variant | seeds | chart | doc | spatial | macro | Δ vs dense16 [95% CI] | p |
|---|---|---|---|---|---|---|---|
| base (NF4) | 1 | 72.0 | 82.2 | 60.5 | 71.6 | −6.8 [−9.7, −4.1] | 1.000 |
| dense16 | 3 | 73.5 ± 1.3 | 83.5 ± 0.8 | 78.2 ± 2.1 | 78.4 ± 0.7 | – | – |
| dense64 | 1 | 76.5 | 83.0 | 75.0 | 78.2 | −0.2 [−2.4, +2.0] | 0.578 |
| moe | 2 | 74.5 ± 0.7 | 83.3 ± 1.8 | 73.2 ± 3.9 | 77.0 ± 0.9 | −1.4 [−2.7, −0.0] | 0.978 |
| moe_seq | 3 | 73.0 ± 1.7 | 84.4 ± 0.5 | 75.8 ± 0.8 | 77.7 ± 0.9 | −0.7 [−1.7, +0.4] | 0.897 |

Per-run test macro: dense16 77.62 / 78.52 / 79.06 · dense64 78.18 · moe 76.37 / 77.68 · moe_seq
76.94 / 78.66 / 77.59.

## Efficiency (TEST, run_efficiency.sh)
| model | macro | chart | doc | spatial | img tok | weights | peak VRAM | latency T4 |
|---|---|---|---|---|---|---|---|---|
| fp16 base @512 | 76.8 | 76.5 | 84.9 | 69.0 | 440.6 | 4213 MiB | 4325 MiB | 0.503 s |
| NF4 base @512 | 71.6 | 72.0 | 82.2 | 60.5 | 440.6 | 1390 MiB | 1566 MiB | 0.557 s |
| NF4 base @256 | 67.3 | 70.5 | 72.8 | 58.5 | 241.7 | 1390 MiB | 1535 MiB | 0.393 s |
| dense16 @256 (3 seeds) | 73.4 ± 0.5 | 70.0 | 74.7 | 75.7 | 241.7 | 1444 MiB | 1588 MiB | ~0.50 s |
| moe_seq @256 (3 seeds) | 74.6 ± 1.0 | 71.7 | 75.0 | 77.0 | 241.7 | 1606 MiB | 1753 MiB | ~0.52 s |

vs fp16 @512: dense16_256 −3.3 [−6.3, −0.4] (p 0.986); moe_seq_256 −2.2 [−5.1, +0.7] (p 0.940).
Per-run @256 test: dense16 73.86 / 73.55 / 72.93 · moe_seq 73.61 / 75.63 / 74.48. Dev bests @256:
dense16 80.79 / 80.12 / 78.29 · moe_seq 81.09 / 79.40 / 76.95. Train time @256: ~36 min (dense16),
~42 min (moe_seq) vs 60-68 min @512.
Reading: our 4-bit models @512 (78.4 / 77.7) exceed the fp16 model (76.8) at ~1.6-1.75 GB vs 4.3 GB peak.

## Edge benchmark, laptop GTX 1650 4 GB (bench.py, NF4 weights, fp32 compute, 30 dev samples)
| variant | weights | peak | TTFT | latency | decode | score* |
|---|---|---|---|---|---|---|
| base | 1840 MiB | 2054 MiB | 2.91 s | 3.09 s | 18.8 tok/s | 0.718 |
| moe_seq_s0 routed | 2056 MiB | 2278 MiB | 3.01 s | 3.25 s | 14.2 tok/s | 0.785 |
| moe_seq_s0 merged | 1840 MiB | 2054 MiB | 2.91 s | 3.08 s | 18.8 tok/s | 0.785 |
| moe_s0 routed | 2056 MiB | 2278 MiB | 3.00 s | 3.27 s | 13.1 tok/s | 0.760 |
| dense16_s0 merged | 1840 MiB | 2054 MiB | 2.90 s | 3.08 s | 18.7 tok/s | 0.787 |

*30 samples, sanity check only. fp16 compute on the GTX 1650 is ~4x slower (base 13.5 s vs 3.4 s;
measured GEMM 0.34 vs 1.57 TFLOPS fp16 vs fp32: no tensor cores). The fp16 model (4.3 GB) does not fit.
CPU-only (i5-12450H, bf16, 8 GB RAM): ~5 min per sample, impractical in PyTorch.
NF4 with an fp16 vision encoder (skip_quant vision): 2339 MiB weights on the GTX 1650.
HQQ (mopeq.py) uniform 4-bit: 1526 MiB weights; a mixed plan at ~3 bits: 1326 MiB.

## Pending
- run_vision.sh (where the 5.2-point NF4 loss comes from; moe_seq / dense16 with fp16 vision).
- run_mopeq.sh (MoPEQ-style sensitivity-guided mixed precision vs uniform HQQ 3/4-bit, TEST).
