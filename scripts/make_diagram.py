"""Method diagram and per-task test bars for the paper (PDF, IEEE column width).  Numbers are copied
from docs/RESULTS_LOG.md (no run outputs needed).  Usage: python scripts/make_diagram.py --out paper/figs
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
COL = 3.5  # IEEE column width, inches
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def fig_per_task(out):
    tasks = ["chart", "document", "spatial"]
    rows = [("fp16 model", [76.5, 84.9, 69.0], None, "#c3c2b7"),
            ("NF4 base", [72.0, 82.2, 60.5], None, BLUE),
            ("dense LoRA (NF4)", [73.5, 83.5, 78.2], [1.3, 0.8, 2.1], AQUA),
            ("MoE-LoRA (NF4)", [73.0, 84.4, 75.8], [1.7, 0.5, 0.8], YELLOW),
            ("MoE-LoRA (HQQ-4)", [76.8, 83.8, 78.2], [0.8, 0.8, 1.6], ORANGE)]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    w = 0.16
    for i, (lab, v, sd, c) in enumerate(rows):
        xs = [t + (i - 2) * (w + 0.008) for t in range(3)]
        ax.bar(xs, v, w, yerr=sd, color=c, label=lab, error_kw=dict(elinewidth=0.6, capsize=1.2, ecolor=INK))
    ax.set_xticks(range(3)); ax.set_xticklabels(tasks)
    ax.set_ylim(55, 90); ax.set_ylabel("test score (%)")
    ax.grid(axis="y", color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False, columnspacing=0.8,
              handlelength=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "per_task.pdf")); plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec=MUTED, fs=6.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, mutation_scale=6))


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(COL * 2, 2.05))
    ax.set_xlim(0, 10); ax.set_ylim(-0.05, 3); ax.axis("off")
    FROZEN, TRAIN, LIGHT = "#e8e7e3", "#fbe1d6", "#cde2fb"
    ax.text(0.05, 2.85, "(a) Training: shared 4-bit base, sequence-level router", fontsize=7.5, fontweight="bold")
    box(ax, 0.05, 1.75, 1.1, 0.7, "image", LIGHT)
    box(ax, 0.05, 0.85, 1.1, 0.7, "question\ntext", LIGHT)
    box(ax, 1.4, 1.75, 1.35, 0.7, "vision encoder\n(4-bit, frozen)", FROZEN)
    box(ax, 1.4, 0.85, 1.35, 0.7, "router: pooled\nprompt -> expert e", TRAIN, fs=6)
    arrow(ax, 1.15, 2.1, 1.4, 2.1); arrow(ax, 1.15, 1.2, 1.4, 1.2)
    ax.add_patch(FancyBboxPatch((3.0, 0.0), 2.9, 2.6, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc="white", ec=MUTED, lw=0.7, ls=(0, (3, 2))))
    ax.text(4.45, 2.42, "language model, 28 layers", ha="center", fontsize=6.5, color=MUTED)
    box(ax, 3.15, 1.65, 2.6, 0.55, "attention (4-bit, frozen)", FROZEN)
    box(ax, 3.15, 0.95, 1.2, 0.55, "MLP W\n(4-bit, frozen)", FROZEN)
    for i in range(4):
        box(ax, 4.45 + i * 0.33, 0.95, 0.28, 0.55, "E%d" % (i + 1), TRAIN if i == 1 else "#fdf3ee", fs=6,
            ec=ORANGE if i == 1 else MUTED)
    ax.text(4.45, 0.2, "y = Wx + B$_e$A$_e$x\nsame expert e for the whole request", ha="center", va="center", fontsize=6)
    arrow(ax, 2.75, 2.1, 3.15, 1.95)
    ax.plot([2.075, 2.075, 4.92], [0.85, 0.62, 0.62], color=ORANGE, lw=0.8)
    ax.annotate("", (4.92, 0.95), (4.92, 0.62),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ORANGE, mutation_scale=6))
    box(ax, 6.05, 1.65, 0.8, 0.55, "answer", LIGHT)
    arrow(ax, 5.75, 1.92, 6.05, 1.92)
    ax.text(6.45, 1.2, "KD from\nfp16 model", ha="center", fontsize=6, color=MUTED)
    ax.plot([7.0, 7.0], [0.1, 2.8], color="#d0cfca", lw=0.6)
    ax.text(7.15, 2.85, "(b) Deployment: fold one expert", fontsize=7.5, fontweight="bold")
    box(ax, 7.2, 1.75, 2.7, 0.65, "W$'$ = W + B$_e$A$_e$\n(one checkpoint per task)", TRAIN, fs=6.3)
    box(ax, 7.2, 0.85, 2.7, 0.65, "same memory and speed as the\n4-bit base (no router, no experts)", FROZEN, fs=6.3)
    arrow(ax, 8.55, 1.75, 8.55, 1.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(out, "architecture.pdf")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_per_task(a.out)
    fig_architecture(a.out)
    print("figures ->", a.out)
