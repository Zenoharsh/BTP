import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, bg='#ffffff', ec='#333333', lw=1.5, fontsize=9, text_color='black', fontweight='normal'):
    box = patches.Rectangle((x, y), w, h, facecolor=bg, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, color=text_color, fontweight=fontweight, zorder=3, linespacing=1.4)

def draw_arrow(ax, x1, y1, x2, y2, color='black', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=color, edgecolor=color, shrink=0.0, width=lw, headwidth=6, headlength=8), zorder=1)

def generate_pro_diagram():
    print("Generating Academic VLM-MoE Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.axis('off')

    # Main Title
    plt.title("ARCHITECTURE", loc='left', fontsize=14, fontweight='bold', pad=20)
    ax.text(0.5, 1.02, "SPARSE UPCYCLED VISION-LANGUAGE Mixture-of-Experts (VLM-MoE)", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a365d')

    # Column Headers (Matching Senior's Layout)
    headers = ["1. INPUT STREAMS", "2. FEATURE ENCODERS", "3. SMART ROUTING LAYER", "4. EDGE DEPLOYMENT"]
    x_positions = [0.1, 0.35, 0.65, 0.9]
    for x, title in zip(x_positions, headers):
        ax.text(x, 0.95, title, ha='center', va='center', fontsize=11, fontweight='bold')

    # Vertical Separator Lines
    for x in [0.22, 0.48, 0.81]:
        ax.plot([x, x], [0.1, 0.93], color='gray', linestyle='--', linewidth=1, zorder=0)

    # --- COLUMN 1: INPUTS ---
    draw_box(ax, 0.02, 0.75, 0.16, 0.12, "Raw Images (DocVQA)\n\nMultimodal Features\n(Spatial, ChartQA)")
    draw_box(ax, 0.02, 0.35, 0.16, 0.12, "Tokenized Prompts\n\nInstruction JSON\n(Chat Template)")

    # --- COLUMN 2: ENCODERS ---
    # ViT Block
    draw_box(ax, 0.25, 0.70, 0.20, 0.18, 
             "Vision Transformer (ViT)\nPatch Extraction & Projection\nMulti-Head Attention", bg='#e6f2ff', ec='#2b6cb0')
    ax.text(0.46, 0.85, "1536-D\nVisual\nEmbeddings\n$(v_i)$", ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Text Block
    draw_box(ax, 0.25, 0.30, 0.20, 0.18, 
             "Qwen2-VL Base\nText Embedding Layer\n(Frozen Weights)", bg='#e6f2ff', ec='#2b6cb0')
    ax.text(0.46, 0.45, "1536-D\nText\nEmbeddings\n$(t_i)$", ha='left', va='center', fontsize=9, fontweight='bold')

    # --- COLUMN 3: SMART ROUTING (THE MATH) ---
    # Gating Network
    draw_box(ax, 0.52, 0.75, 0.25, 0.12, "", bg='#f7fafc')
    ax.text(0.645, 0.85, "Top-2 Gating Network", ha='center', va='center', fontweight='bold')
    draw_box(ax, 0.53, 0.755, 0.23, 0.08, r"$P(x) = \mathrm{Softmax}(\mathbf{W}_g \cdot x)$", lw=0)
    
    # Mathematical Losses (Red text like senior's diagram)
    ax.text(0.645, 0.69, r"$L_{aux} = \alpha \cdot L_{bal} + \beta \cdot \log^2(\sum e^{logits})$", 
            ha='center', va='center', fontsize=10, color='#c53030', fontweight='bold')
    ax.text(0.645, 0.64, "IF $L_{aux} > \\epsilon$ : Apply Z-Loss Penalty\nIF Active Experts < 2 : Drop Token", 
            ha='center', va='center', fontsize=9, color='#c53030')

    # Expert SwiGLU Block
    draw_box(ax, 0.52, 0.40, 0.25, 0.20, "", bg='#2d3748')
    ax.text(0.645, 0.575, "SwiGLU Domain Experts (N=4)", ha='center', va='center', color='white', fontweight='bold')
    draw_box(ax, 0.53, 0.41, 0.23, 0.15, 
             r"$\mathrm{Swish}(x\mathbf{W}_{gate}) \odot x\mathbf{W}_{up}$" + "\n" + r"$\downarrow$" + "\n" + r"$(\dots) \mathbf{W}_{down}$", 
             bg='white', lw=0)
    ax.text(0.78, 0.5, "5504-D\nIntermediate\nState", ha='left', va='center', fontsize=8, rotation=-90)

    # Fusion / Recombination Block
    draw_box(ax, 0.52, 0.15, 0.25, 0.15, "MoE Fusion Block\n\n" + r"$H_{out} = \sum_{i \in \mathrm{Top2}} P(x)_i \cdot E_i(x)$", bg='#f7fafc')
    ax.text(0.645, 0.10, "Reconstructed 1536-D Joint Embedding", ha='center', va='center', fontsize=9, fontweight='bold')

    # --- COLUMN 4: EDGE OUTPUT ---
    draw_box(ax, 0.84, 0.70, 0.15, 0.12, "MoPEQ Quantization\n\nMixed Precision\n(4-bit & 8-bit)")
    draw_box(ax, 0.84, 0.50, 0.15, 0.14, "Hardware Target\n\nRaspberry Pi 5\nJetson Nano\n< 4GB VRAM limit")
    
    # PyTorch Snippet Box
    code_text = "def forward(self, x):\n  gate = self.router(x)\n  idx = gate.topk(2)\n  return self.moe(x, idx)"
    draw_box(ax, 0.83, 0.20, 0.17, 0.18, code_text, bg='#1e1e1e', text_color='#00ffcc', fontsize=8)
    ax.text(0.915, 0.40, "PyTorch Implementation", ha='center', va='center', fontsize=9, fontweight='bold')

    # --- ARROWS ---
    # Inputs to Encoders
    draw_arrow(ax, 0.18, 0.81, 0.25, 0.81)
    draw_arrow(ax, 0.18, 0.41, 0.25, 0.41)
    
    # Encoders to Router
    draw_arrow(ax, 0.45, 0.79, 0.52, 0.81)
    draw_arrow(ax, 0.45, 0.39, 0.52, 0.78)
    
    # Router to Experts to Fusion
    draw_arrow(ax, 0.645, 0.75, 0.645, 0.60)
    draw_arrow(ax, 0.645, 0.40, 0.645, 0.30)
    
    # Fusion to Output
    draw_arrow(ax, 0.77, 0.22, 0.81, 0.22, color='#c53030') # Red output arrow matching senior's
    draw_arrow(ax, 0.81, 0.22, 0.81, 0.76)
    draw_arrow(ax, 0.81, 0.76, 0.84, 0.76)
    draw_arrow(ax, 0.81, 0.57, 0.84, 0.57)

    plt.tight_layout()
    plt.savefig("VLM_Architecture_Pro.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[✔] Academic architecture diagram saved as VLM_Architecture_Pro.png")

if __name__ == "__main__":
    generate_pro_diagram()