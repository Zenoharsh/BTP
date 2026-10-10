"""
Paper figures (vector PDF, IEEE column widths) -> paper/figs/.

    python scripts/make_figures.py            # all figures whose inputs exist under runs/

Inputs: runs/mopeq/{sens_kl_*,sens_hessian_*,plan_*}.json, runs/bench/gtx1650_*_rerun.json and the
TEST numbers below (peak memory is not stored in the prediction files; values are copied from the
eval logs, see docs/RESULTS_LOG.md). Macro scores are cross-checked against runs/*/test_preds.jsonl
whenever those files are present.
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "paper/figs"
COL, FULL = 3.5, 7.16                          # IEEE single / double column width (in)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
GRAY, INK, MUTED = "#8a8986", "#0b0b0b", "#52514e"
SEQ = {2: "#86b6ef", 3: "#3987e5", 4: "#1c5cab", 8: "#0d366b"}   # ordinal blue ramp (250/400/550/700)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# --------------------------------------------------------------------------- data
# (label, macro, std over seeds or None, peak GPU MiB, group, prediction-file glob for the check)
ACC_MEM = [
    ("fp16 base", 76.8, None, 4325, "base", "runs/fp16_512/test_preds.jsonl"),
    ("NF4 base", 71.6, None, 1566, "base", "runs/b0_512/test_preds.jsonl"),
    ("NF4 vision + fp16 LM", 74.8, None, 3411, "base", "runs/b0lang_512/test_preds.jsonl"),
    ("fp16 vision + NF4 LM", 73.3, None, 2481, "base", "runs/b0vis_512/test_preds.jsonl"),
    ("HQQ 4-bit", 74.0, None, 1712, "base", "runs/hqq_u4/test_preds.jsonl"),
    ("HQQ 3-bit", 66.8, None, 1508, "base", "runs/hqq_u3/test_preds.jsonl"),
    ("MoPEQ-KL 4", 75.2, None, 1716, "mixed", "runs/hqq_kl_4/test_preds.jsonl"),
    ("MoPEQ-KL 3.5", 74.0, None, 1622, "mixed", "runs/hqq_kl_3.5/test_preds.jsonl"),
    ("MoPEQ-KL 3", 69.0, None, 1506, "mixed", "runs/hqq_kl_3/test_preds.jsonl"),
    ("MoE-LoRA, HQQ 4-bit", 79.6, 0.8, 1909, "adapt", "runs/moe_seq_hqq4_s*/test_preds.jsonl"),
    ("MoE-LoRA, MoPEQ-KL 3.5", 78.0, 0.7, 1815, "adapt", "runs/moe_seq_mopeq35_s*/test_preds.jsonl"),
    ("MoE-LoRA, fp16 vision", 78.0, 0.8, 2700, "adapt", "runs/moe_seq_vis_s*/test_preds.jsonl"),
    ("LoRA, fp16 vision", 78.6, 0.7, 2535, "adapt", "runs/dense16_vis_s*/test_preds.jsonl"),
    ("MoE-LoRA, NF4, 256 px", 74.6, 1.0, 1753, "adapt", None),
    ("LoRA, NF4, 256 px", 73.4, 0.5, 1588, "adapt", None),
]
TASKS = ("chart_qa", "document_ocr", "spatial_reasoning")


def macro_of(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return 100 * np.mean([np.mean([r["score"] for r in rows if r["task"] == t]) for t in TASKS])


def check_scores():
    for label, macro, _, _, _, pat in ACC_MEM:
        files = sorted(glob.glob(pat)) if pat else []
        if files:
            m = np.mean([macro_of(f) for f in files])
            assert abs(m - macro) < 0.06, f"{label}: table {macro} vs predictions {m:.2f}"
    print("scores cross-checked against prediction files")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    plt.close(fig)
    print("->", os.path.join(OUT, name + ".pdf"))


# --------------------------------------------------------------------------- figures
def fig_acc_memory():
    style = {"base": dict(c=GRAY, m="o", fill=False, name="base, no adapters"),
             "mixed": dict(c=ORANGE, m="D", fill=True, name="MoPEQ-KL plan, no adapters"),
             "adapt": dict(c=BLUE, m="s", fill=True, name="with adapters (±1 s.d., 3 seeds)")}
    # label anchor in data coordinates (x MiB, y score, horizontal alignment); a leader line is drawn
    # whenever the label sits away from its point
    place = {
        "fp16 base": (4280, 75.9, "right"), "NF4 base": (1500, 71.6, "right"),
        "NF4 vision + fp16 LM": (3480, 74.8, "left"), "fp16 vision + NF4 LM": (2550, 72.7, "left"),
        "HQQ 4-bit": (2020, 73.6, "left"), "HQQ 3-bit": (1440, 66.8, "right"),
        "MoPEQ-KL 4": (2020, 76.1, "left"), "MoPEQ-KL 3.5": (1500, 75.1, "right"),
        "MoPEQ-KL 3": (1440, 69.0, "right"), "MoE-LoRA, HQQ 4-bit": (2020, 80.4, "left"),
        "MoE-LoRA, MoPEQ-KL 3.5": (1700, 79.2, "right"), "MoE-LoRA, fp16 vision": (2780, 77.5, "left"),
        "LoRA, fp16 vision": (2600, 79.3, "left"), "MoE-LoRA, NF4, 256 px": (2020, 74.9, "left"),
        "LoRA, NF4, 256 px": (1500, 73.1, "right"),
    }
    fig, ax = plt.subplots(figsize=(COL, 2.85))
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.axvline(4096, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(4140, 70.8, "4 GB laptop GPU", rotation=90, ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.axhline(76.8, color=GRAY, lw=0.6, ls=(0, (1, 1.5)))
    for g, s in style.items():
        pts = [r for r in ACC_MEM if r[4] == g]
        for label, y, sd, x, _, _ in pts:
            if sd:
                ax.errorbar(x, y, yerr=sd, color=s["c"], lw=0.8, capsize=0, zorder=2)
            ax.scatter(x, y, s=22, marker=s["m"], facecolors=s["c"] if s["fill"] else "white",
                       edgecolors="white" if s["fill"] else s["c"], linewidths=1.2 if s["fill"] else 0.9,
                       zorder=3, label=s["name"] if label == pts[0][0] else None)
            tx, ty, ha = place[label]
            far = abs(tx - x) > 140 or abs(ty - y) > 0.45
            text = label.replace(" + ", ",\n") if "vision + " in label else label
            ax.annotate(text, (x, y), xytext=(tx, ty), textcoords="data", ha=ha, va="center", fontsize=6,
                        color=INK, zorder=4,
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkA=1, shrinkB=3) if far else None)
    ax.set_xlabel("Peak GPU memory during evaluation (MiB)")
    ax.set_ylabel("Test macro score")
    ax.set_xlim(900, 4600)
    ax.set_ylim(66, 81)
    ax.legend(loc="lower left", bbox_to_anchor=(0.27, 0.0), frameon=False, handletextpad=0.3,
              borderaxespad=0.3)
    save(fig, "acc_vs_memory")


def load_sens(pattern):
    sizes, sens = {}, {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        sizes.update(d["sizes"])
        sens.update(d["sens"])
    return sizes, sens


def unit_order(units):
    vis = sorted([u for u in units if u.startswith("vis.") and u != "vis.merger"], key=lambda u: int(u.split(".")[1]))
    lm = sorted([u for u in units if u.startswith("lm.")], key=lambda u: (int(u.split(".")[1]), u.endswith("mlp")))
    return vis + (["vis.merger"] if "vis.merger" in units else []) + lm


def fig_mopeq_bits():
    sizes, kl = load_sens("runs/mopeq/sens_kl_[0-9].json")
    plans = [("uniform 4", "plan_u4"), ("KL 4", "plan_kl_4"), ("KL 3.5", "plan_kl_3.5"), ("KL 3", "plan_kl_3"),
             ("Hessian 4", "plan_hess_4")]
    plans = [(n, json.load(open(f"runs/mopeq/{p}.json"))["bits"]) for n, p in plans
             if os.path.exists(f"runs/mopeq/{p}.json")]
    if not kl or not plans:
        print("skip mopeq_bits (inputs missing)")
        return
    order = unit_order(kl)
    n_vis = sum(u.startswith("vis.") and u != "vis.merger" for u in order)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(FULL, 2.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.08})
    x = np.arange(len(order))
    s3 = np.array([kl[u]["3"] for u in order])
    colors = [ORANGE if u.startswith("vis") else BLUE for u in order]
    a1.bar(x, s3, width=0.75, color=colors, linewidth=0)
    a1.set_yscale("log")
    a1.set_ylabel("KL at 3 bit", labelpad=2)
    a1.grid(True, axis="y")
    a1.set_axisbelow(True)
    for u in sorted(order, key=lambda u: -kl[u]["3"])[:5]:
        i = order.index(u)
        a1.annotate(u.replace("vis.merger", "merger").replace("vis.", "V").replace("lm.", "L")
                    .replace(".attn", " attn").replace(".mlp", " MLP"),
                    (i, kl[u]["3"]), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.5, color=INK)
    a1.set_ylim(top=s3.max() * 3)
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["vision encoder", "language model"], loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, ncol=2)
    for r, (name, bits) in enumerate(plans):
        for i, u in enumerate(order):
            a2.add_patch(plt.Rectangle((i - 0.5, r - 0.42), 1, 0.84, color=SEQ[int(bits[u])], linewidth=0))
    a2.set_ylim(len(plans) - 0.5, -0.5)
    a2.set_yticks(range(len(plans)))
    a2.set_yticklabels([n for n, _ in plans])
    a2.set_xlim(-0.6, len(order) - 0.4)
    a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", length=0)
    ticks = [n_vis / 2 - 0.5] + [n_vis + 1 + 2 * k + 0.5 for k in range(0, 28, 4)]
    a2.set_xticks(ticks)
    a2.set_xticklabels(["vision blocks 0-31 and merger"] + [f"L{k}" for k in range(0, 28, 4)])
    for ax in (a1, a2):
        ax.axvline(n_vis + 0.5, color=MUTED, lw=0.6)
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SEQ[b]) for b in (2, 3, 4, 8)],
              labels=[f"{b} bit" for b in (2, 3, 4, 8)], loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=4, frameon=False)
    save(fig, "mopeq_bits")


def fig_kl_vs_hessian():
    _, kl = load_sens("runs/mopeq/sens_kl_[0-9].json")
    _, he = load_sens("runs/mopeq/sens_hessian_[0-9].json")
    us = sorted(set(kl) & set(he))
    if not us:
        print("skip kl_vs_hessian (inputs missing)")
        return
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    floor = min(v["3"] for v in he.values() if v["3"] > 0) / 3
    for part, c, name in (("vis", ORANGE, "vision encoder"), ("lm", BLUE, "language model")):
        sel = [u for u in us if u.startswith(part)]
        xs = [kl[u]["3"] for u in sel]
        ys = [max(he[u]["3"], floor) for u in sel]
        ax.scatter(xs, ys, s=12, color=c, edgecolors="white", linewidths=0.4, label=name, zorder=3)
    zero = sum(he[u]["3"] <= 0 for u in us)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    ax.set_xlabel("measured KL sensitivity at 3 bit")
    ax.set_ylabel("Hessian-trace sensitivity at 3 bit")
    rho = spearman([kl[u]["3"] for u in us], [he[u]["3"] for u in us])
    note = f"Spearman ρ = {rho:.2f} ({len(us)} units)"
    if zero:
        note += f"\n{zero} units with trace ≤ 0 (shown at the floor)"
    ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "kl_vs_hessian")


def spearman(a, b):
    """Tie-aware Spearman rho (average ranks), same as scripts/mopeq_compare.py."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def fig_edge():
    f1, f2 = "runs/bench/gtx1650_moe_seq_s0_rerun.json", "runs/bench/gtx1650_moe_s0_rerun.json"
    if not os.path.exists(f1):
        print("skip edge (bench results missing)")
        return
    r = {d["variant"]: d for d in json.load(open(f1))}
    rows = [("base model", r["base"], GRAY), ("MoE-LoRA, routed", r["routed"], ORANGE),
            ("MoE-LoRA, folded", r["merged"], BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.45), gridspec_kw={"wspace": 0.55})
    for ax, key, xl, fmt in ((axes[0], "peak_mib", "peak GPU memory (MiB)", "{:.0f}"),
                             (axes[1], "decode_tps", "decode speed (tokens/s)", "{:.1f}")):
        vals = [d[key] for _, d, _ in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, height=0.62, color=[c for *_, c in rows], linewidth=0)
        for yi, v in zip(y, vals):
            ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", fontsize=6.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _, _ in rows] if ax is axes[0] else [])
        ax.set_xlabel(xl)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
    save(fig, "edge_gtx1650")


def fig_routing():
    """Per-layer MI(task; expert), token vs sequence routing (seed 0), and the task->expert shares at
    the most task-specific layer of sequence routing."""
    import sys
    sys.path.insert(0, os.path.abspath("."))
    from convert import routing_stats
    runs = [("token routing (moe)", "runs/moe_s0/routing_dev.npz", GRAY),
            ("sequence routing (moe_seq)", "runs/moe_seq_s0/routing_dev.npz", BLUE)]
    runs = [(n, routing_stats(f), c) for n, f, c in runs if os.path.exists(f)]
    if len(runs) < 2:
        print("skip routing (routing_dev.npz missing)")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 1.75), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.28})
    for name, st, c in runs:
        mi = st["mi"]
        a1.plot(np.arange(len(mi)), mi, color=c, lw=1.5, marker="o", ms=3, label=f"{name}, mean {mi.mean():.2f}")
    a1.axhline(runs[0][1]["max_mi"], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    a1.text(27, runs[0][1]["max_mi"], "maximum = log2 3", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    a1.set_xlabel("language layer")
    a1.set_ylabel("MI(task; expert) (bits)")
    a1.set_xlim(-0.5, 27.5)
    a1.set_ylim(0, 1.7)
    a1.grid(True, axis="y")
    a1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=2)
    st = runs[1][1]
    l = int(np.argmax(st["mi"]))
    tasks = st["tasks"]
    m = np.stack([st["f_task"][t][l] for t in tasks])
    a2.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#0d366b"]),
              vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            a2.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if m[i, j] > 0.55 else INK)
    a2.set_xticks(range(m.shape[1]))
    a2.set_xticklabels([f"e{j}" for j in range(m.shape[1])])
    a2.set_yticks(range(len(tasks)))
    a2.set_yticklabels([t.replace("_", " ").replace("document ocr", "document").replace("spatial reasoning", "spatial")
                        .replace("chart qa", "chart") for t in tasks])
    a2.set_title(f"sequence routing, layer {l}: share of each task per expert", fontsize=7)
    a2.tick_params(length=0)
    for sp in a2.spines.values():
        sp.set_visible(False)
    save(fig, "routing_mi")


if __name__ == "__main__":
    check_scores()
    fig_acc_memory()
    fig_mopeq_bits()
    fig_kl_vs_hessian()
    fig_edge()
    fig_routing()
