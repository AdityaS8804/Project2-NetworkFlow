#!/usr/bin/env python3
"""Generate tall, vertical presentation diagrams for GNN-BERT Network IDS project."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib import patheffects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "diagrams")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "images")
DPI = 300

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    'card_1': '#F5F0FF', 'card_2': '#EDE5FB', 'card_3': '#E2D4F7',
    'card_4': '#D4BFF2', 'card_5': '#C4A8EC', 'card_accent': '#A78BFA',
    'border': '#7C3AED', 'border_soft': '#A78BFA',
    'text_h1': '#4C1D95', 'text_body': '#3B1272', 'text_muted': '#6D28D9',
    'text_on_dark': '#F5F3FF', 'white': '#FFFFFF',
    'chart_1': '#7C3AED', 'chart_2': '#6366F1', 'chart_mark': '#10B981',
    'warmup_fill': '#DDD6FE',
}

CMAP = LinearSegmentedColormap.from_list(
    'lav', ['#F0E6FF', '#DDD6FE', '#C4B5FD', '#A78BFA', '#8B5CF6'])

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica Neue', 'Arial'],
})

FS_TITLE = 18;  FS_HEAD = 12;  FS_BODY = 10;  FS_SMALL = 8.5;  FS_TINY = 7.5
RND = 0.06;  PAD = 0.015


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gradient_bg(ax, alpha=0.26):
    n = 512
    X, Y = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
    ax.imshow((X+(1-Y))/2, extent=ax.get_xlim()+ax.get_ylim(),
              aspect='auto', cmap=CMAP, alpha=alpha, zorder=0, interpolation='bilinear')

def _lerp(c1, c2, t):
    a, b = to_rgb(c1), to_rgb(c2)
    return tuple(a[i]+(b[i]-a[i])*t for i in range(3))

def _depth(i, n):
    return _lerp(C['card_1'], C['card_5'], i/max(n-1,1))

def box(ax, x, y, w, h, text='', *, fs=FS_BODY, fill=None, border=None,
        tc=None, bold=False, alpha=0.93, shadow=True, zorder=3):
    fill   = fill   or C['card_1']
    border = border or C['border']
    tc     = tc     or C['text_body']
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x+0.04, y-0.04), w, h,
            boxstyle=f"round,pad={PAD},rounding_size={RND}",
            facecolor='#7C3AED', edgecolor='none', alpha=0.08, zorder=zorder-1))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size={RND}",
        facecolor=fill, edgecolor=border, linewidth=1.5, alpha=alpha, zorder=zorder))
    if text:
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
                color=tc, weight='bold' if bold else 'normal',
                linespacing=1.35, zorder=zorder+1)

def arrow(ax, x1, y1, x2, y2, *, color=None, lw=1.8):
    color = color or C['border_soft']
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=12), zorder=2)

def curved_arrow(ax, x1, y1, x2, y2, *, color=None, lw=1.8, rad=0.25):
    color = color or C['border_soft']
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=12,
                                connectionstyle=f'arc3,rad={rad}'), zorder=2)

def label(ax, x, y, text, *, fs=FS_SMALL, color=None, ha='left'):
    ax.text(x, y, text, fontsize=fs, color=color or C['text_muted'],
            style='italic', ha=ha, va='center', zorder=5)

def heading(ax, x, y, text, *, fs=FS_TITLE):
    ax.text(x, y, text, fontsize=fs, color=C['text_h1'], ha='center',
            va='center', weight='bold', zorder=10,
            path_effects=[patheffects.withSimplePatchShadow(
                offset=(1.2,-1.2), shadow_rgbFace=C['card_4'], alpha=0.30)])

def save(fig, name):
    p = os.path.join(OUTPUT_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"  -> {p}")

def canvas(W, H):
    fig, ax = plt.subplots(figsize=(W, H))
    fig.set_facecolor(C['white'])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    _gradient_bg(ax)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
#  Slide 5 — Data Pipeline  (10 × 14)
# ══════════════════════════════════════════════════════════════════════════════

def generate_slide05():
    W, H = 10, 14
    fig, ax = canvas(W, H)

    bw, bh = 5.8, 1.15
    bx = 0.8
    gap = 0.55
    mid_x = bx + bw / 2

    heading(ax, mid_x, H-0.7, "Data Pipeline")
    ax.text(mid_x, H-1.25, "Flow Processing & Graph Construction",
            fontsize=FS_HEAD, ha='center', color=C['text_muted'], zorder=10)

    stages = [
        "CIC-IDS2017 Raw CSV\n3.1M flows   ·   7 CSV files",
        "Timestamp Cleaning\n2.83M valid flows   ·   288K dropped",
        "Sliding Window\n30s window   ·   10s stride",
        "Graph Construction\nNodes = IPs   ·   Edges = Flows\n77 features / edge",
        "PyG Data Objects\nLouvain   ·   Degree   ·   Clustering coeff.",
    ]

    top_y = H - 2.2
    ys = [top_y - i * (bh + gap) for i in range(len(stages))]

    for i, (txt, by) in enumerate(zip(stages, ys)):
        box(ax, bx, by, bw, bh, txt, fs=FS_BODY,
            fill=_depth(i, len(stages)), bold=(i == 0))

    for i in range(len(stages)-1):
        arrow(ax, mid_x, ys[i], mid_x, ys[i+1]+bh)

    # Side annotations
    ax_ = bx + bw + 0.3
    annots = [
        ((ys[0]+ys[1]+bh)/2, "9.3 % dropped\n(timestamp)"),
        ((ys[1]+ys[2]+bh)/2, "4,879 candidates"),
        ((ys[2]+ys[3]+bh)/2, "2,934 valid graphs\n(≥ 3 nodes)"),
        ((ys[3]+ys[4]+bh)/2, "15 types → 7 classes"),
    ]
    for ay_, txt in annots:
        label(ax, ax_, ay_, txt, fs=FS_TINY)

    # 7-class table card just below last box
    cw, ch = 5.5, 2.1
    cx = (W - cw) / 2
    cy = ys[-1] - 0.6 - ch
    box(ax, cx, cy, cw, ch, '', fill=C['card_1'], alpha=0.90)
    ax.text(cx+cw/2, cy+ch-0.22, "7 Attack Classes",
            fontsize=FS_HEAD, ha='center', color=C['text_h1'], weight='bold', zorder=10)
    classes = [("Benign","2.27M"),("DoS","253K"),("DDoS","128K"),
               ("PortScan","159K"),("BruteForce","14K"),("WebAttack","2.2K"),("Bot/Other","2.0K")]
    for j, (nm, ct) in enumerate(classes):
        yy = cy + ch - 0.52 - j * 0.22
        ax.text(cx+0.4, yy, f"{j}  {nm}", fontsize=FS_TINY, color=C['text_body'],
                family='monospace', zorder=10)
        ax.text(cx+cw-0.4, yy, ct, fontsize=FS_TINY, color=C['text_muted'],
                family='monospace', ha='right', zorder=10)

    save(fig, "slide05_data_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Slide 7 — Stage 1 GNN  (10 × 16)
# ══════════════════════════════════════════════════════════════════════════════

def generate_slide07():
    W, H = 10, 16
    fig, ax = canvas(W, H)
    heading(ax, W/2, H-0.7, "Stage 1: GATv2Conv Encoder")
    ax.text(W/2, H-1.25, "Graph Classification & Training",
            fontsize=FS_HEAD, ha='center', color=C['text_muted'], zorder=10)

    # ── Architecture column (stacked vertically, centered) ──
    bw, bh = 7.5, 0.75
    bx = (W - bw) / 2
    gap = 0.35

    layers = [
        ("Input  [N, 77]",                                    C['card_1']),
        ("GATv2Conv (77→128, 4 heads) + ELU  →  [N, 512]",   C['card_2']),
        ("GATv2Conv (512→128, 4 heads) + ELU  →  [N, 512]",  C['card_2']),
        ("GATv2Conv (512→128, 1 head)  →  [N, 128]",         C['card_3']),
        ("global_mean_pool  →  [B, 128]",                     C['card_3']),
        ("MLP :  128 → 64 → 7 logits",                       C['card_4']),
    ]

    top_y = H - 2.0
    ys = [top_y - i * (bh + gap) for i in range(len(layers))]

    for (txt, fill), by in zip(layers, ys):
        box(ax, bx, by, bw, bh, txt, fs=FS_BODY, fill=fill)

    for i in range(len(layers)-1):
        arrow(ax, W/2, ys[i], W/2, ys[i+1]+bh)

    # ── Training config card (below architecture) ──
    cfg_y = ys[-1] - 0.4 - 1.6
    cfg_w, cfg_h = 7.5, 1.6
    cfg_x = (W - cfg_w) / 2
    box(ax, cfg_x, cfg_y, cfg_w, cfg_h, '', fill=C['card_1'], alpha=0.90)
    ax.text(cfg_x+cfg_w/2, cfg_y+cfg_h-0.2, "Training Configuration",
            fontsize=FS_HEAD, ha='center', color=C['text_h1'], weight='bold', zorder=10)
    cfg = ("Parameters:  747,527  (738K encoder + 9K head)\n"
           "Split:  2,053 train  /  440 val  /  441 test\n"
           "Oversampled:  12,096 balanced graphs\n"
           "Epochs: 20    ·    Loss: Cross-Entropy")
    ax.text(cfg_x+cfg_w/2, cfg_y+cfg_h/2-0.15, cfg, fontsize=FS_BODY,
            ha='center', va='center', color=C['text_body'], linespacing=1.4, zorder=10)

    # ── t-SNE image card (bottom) ──
    tsne_y = 0.3
    tsne_h = cfg_y - 0.5 - tsne_y
    tsne_w = 8.0
    tsne_x = (W - tsne_w) / 2
    box(ax, tsne_x, tsne_y, tsne_w, tsne_h, '', fill=C['card_1'], alpha=0.82)
    ax.text(tsne_x+tsne_w/2, tsne_y+tsne_h-0.22, "t-SNE: 7-Class Cluster Separation",
            fontsize=FS_HEAD, ha='center', color=C['text_h1'], weight='bold', zorder=10)

    tsne_path = os.path.join(RESULTS_DIR, "stage1", "tsne_attack_classes.png")
    if os.path.exists(tsne_path):
        img = Image.open(tsne_path)
        wi, hi = img.size
        img = img.crop((0, int(hi*0.16), wi, hi))
        target_h_inches = (tsne_h - 0.8) / H * 16
        img_h_inches = img.size[1] / DPI
        zoom = target_h_inches / img_h_inches * DPI / fig.dpi
        zoom = min(zoom, 0.46)
        im = OffsetImage(np.array(img), zoom=zoom)
        ab = AnnotationBbox(im, (tsne_x+tsne_w/2, tsne_y+tsne_h/2-0.35),
                            frameon=False, zorder=8)
        ax.add_artist(ab)

    save(fig, "slide07_stage1_training.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Slide 8 — QFormer Bridge  (10 × 16)
# ══════════════════════════════════════════════════════════════════════════════

def generate_slide08():
    W, H = 10, 16
    fig, ax = canvas(W, H)
    heading(ax, W/2, H-0.7, "Stage 2: QFormer Bridge")
    ax.text(W/2, H-1.25, "Cross-Modal Graph–Language Alignment",
            fontsize=FS_HEAD, ha='center', color=C['text_muted'], zorder=10)

    bh = 0.75
    col_w = 4.0
    col_gap = 0.6
    lx = (W - 2*col_w - col_gap) / 2
    rx = lx + col_w + col_gap

    # ── GNN path (left) ──
    row = H - 2.2

    box(ax, lx, row, col_w, bh, "Frozen GNN Encoder\n128-dim node emb.",
        fs=FS_BODY, fill=C['card_2'], bold=True)
    arrow(ax, lx+col_w/2, row, lx+col_w/2, row-bh-0.12)
    row -= bh + 0.35

    box(ax, lx, row, col_w, bh, "Input Projection\nLinear (128 → 256)",
        fs=FS_BODY, fill=C['card_2'])
    arrow(ax, lx+col_w/2, row, lx+col_w/2, row-bh-0.12)
    row -= bh + 0.35

    # QFormer hero card (spans left column, taller)
    qf_h = 2.6
    qf_x = lx - 0.15
    qf_w = col_w + 0.3
    box(ax, qf_x, row - qf_h + bh, qf_w, qf_h, '', fill=C['card_4'],
        border=C['chart_1'], alpha=0.96)
    qf_top = row + bh - 0.1
    ax.text(qf_x+qf_w/2, qf_top - 0.25, "QFormer Bridge",
            fontsize=13, ha='center', color=C['text_h1'], weight='bold', zorder=10)
    qf = ("4 Learnable Query Tokens\n"
          "2 Cross-Attention Layers (4 heads)\n"
          "256-dim hidden  ·  FFN + LayerNorm\n"
          "Output: mean(queries) → [B, 256]")
    ax.text(qf_x+qf_w/2, qf_top - qf_h/2 - 0.05, qf, fontsize=FS_BODY,
            ha='center', va='center', color=C['text_body'], linespacing=1.4, zorder=10)
    qf_bot = row - qf_h + bh
    arrow(ax, lx+col_w/2, qf_bot, lx+col_w/2, qf_bot-0.35)
    row = qf_bot - 0.35 - bh

    box(ax, lx, row, col_w, bh, "Graph Proj MLP\n(256 → 256 → 256)",
        fs=FS_BODY, fill=C['card_3'])
    arrow(ax, lx+col_w/2, row, lx+col_w/2, row-0.35)
    row -= 0.35 + bh

    l2_graph_y = row
    box(ax, lx, row, col_w, bh, "L2 Normalize\nGraph emb. [B, 256]",
        fs=FS_BODY, fill=C['card_5'], bold=True)

    # ── BERT path (right) ──
    row_r = H - 2.2

    box(ax, rx, row_r, col_w, bh, "Frozen BERT Encoder\n768-dim CLS token",
        fs=FS_BODY, fill=C['card_2'], bold=True)
    arrow(ax, rx+col_w/2, row_r, rx+col_w/2, row_r-bh-0.12)
    row_r -= bh + 0.35

    box(ax, rx, row_r, col_w, bh, "Text Proj MLP\n(768 → 256 → 256)",
        fs=FS_BODY, fill=C['card_2'])

    # Params card on the right (between text proj and L2)
    pm_y = row_r - 0.5 - 1.8
    box(ax, rx, pm_y, col_w, 1.8, '', fill=C['card_1'], alpha=0.92)
    ax.text(rx+col_w/2, pm_y+1.55, "Model Parameters",
            fontsize=FS_HEAD, ha='center', color=C['text_h1'], weight='bold', zorder=10)
    ax.text(rx+col_w/2, pm_y+0.7,
            "Total:  2.83 M\nTrainable:  2.09 M\nQFormer:  1.61 M",
            fontsize=FS_BODY, ha='center', va='center',
            color=C['text_body'], linespacing=1.4, zorder=10)

    # Long arrow from text proj to L2 normalize
    arrow(ax, rx+col_w/2, row_r, rx+col_w/2, l2_graph_y+bh*0.5-0.15)

    box(ax, rx, l2_graph_y, col_w, bh, "L2 Normalize\nText emb. [B, 256]",
        fs=FS_BODY, fill=C['card_5'], bold=True)

    # ── Shared space at bottom ──
    sw = 8.0
    sx = (W - sw) / 2
    sy = l2_graph_y - 1.0
    box(ax, sx, sy, sw, 0.8,
        "Shared 256-dim Space  ·  SigLIP Contrastive Loss",
        fs=FS_HEAD, fill=C['card_accent'], bold=True,
        tc=C['text_on_dark'], border=C['chart_1'])

    curved_arrow(ax, lx+col_w/2, l2_graph_y, sx+sw*0.3, sy+0.8, rad=0.10, lw=2.0)
    curved_arrow(ax, rx+col_w/2, l2_graph_y, sx+sw*0.7, sy+0.8, rad=-0.10, lw=2.0)

    save(fig, "slide08_qformer_bridge.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Slide 11 — Stage 2 Training Results  (10 × 14)
# ══════════════════════════════════════════════════════════════════════════════

def generate_slide11():
    W, H = 10, 14
    fig = plt.figure(figsize=(W, H))
    fig.set_facecolor(C['white'])

    # Background
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0,1); ax_bg.set_ylim(0,1); ax_bg.axis('off')
    n = 512
    X, Y = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
    ax_bg.imshow((X+(1-Y))/2, extent=[0,1,0,1], aspect='auto',
                 cmap=CMAP, alpha=0.22, zorder=0, interpolation='bilinear')

    # Title
    ax_bg.text(0.5, 0.96, "Stage 2: Training Results",
               fontsize=FS_TITLE, ha='center', va='center',
               color=C['text_h1'], weight='bold', zorder=10,
               path_effects=[patheffects.withSimplePatchShadow(
                   offset=(1.2,-1.2), shadow_rgbFace=C['card_4'], alpha=0.30)])

    # Data
    epochs     = list(range(1, 11))
    train_loss = [0.5748, 0.4977, 0.4654, 0.4420, 0.4250, 0.4100, 0.3980, 0.3900, 0.3850, 0.3870]
    val_loss   = [0.5642, 0.5557, 0.5335, 0.5200, 0.5080, 0.4950, 0.4870, 0.4820, 0.4790, 0.4810]
    train_acc  = [4.5, 10.0, 14.0, 18.5, 23.0, 27.0, 30.5, 33.0, 35.0, 34.5]
    val_acc    = [4.2,  6.0,  6.9,  8.5, 10.5, 12.5, 14.0, 15.5, 16.5, 16.0]

    def _style(a):
        a.patch.set_facecolor('white'); a.patch.set_alpha(0.75)
        for s in ('top','right'): a.spines[s].set_visible(False)
        for s in ('left','bottom'):
            a.spines[s].set_color(C['border_soft']); a.spines[s].set_linewidth(0.8)
        a.tick_params(colors=C['text_muted'], labelsize=FS_SMALL)
        a.grid(alpha=0.15, color=C['border_soft']); a.set_xticks(epochs)

    # ── Loss chart (top) ──
    ax1 = fig.add_axes([0.12, 0.56, 0.80, 0.32])
    _style(ax1)
    ax1.axvspan(1, 5, alpha=0.10, color=C['warmup_fill'], label='Warmup')
    ax1.plot(epochs, train_loss, 'o-', color=C['chart_1'], lw=2.5, ms=5, label='Train Loss', zorder=5)
    ax1.plot(epochs, val_loss, 's--', color=C['chart_2'], lw=2.0, ms=4, label='Val Loss', zorder=5)
    ax1.axvline(9, color=C['chart_mark'], ls=':', lw=1.8, alpha=0.85)
    ax1.annotate('Best R@1', xy=(9, train_loss[8]), xytext=(6.5, 0.555),
                 fontsize=FS_BODY, color=C['chart_mark'], weight='bold',
                 arrowprops=dict(arrowstyle='->', color=C['chart_mark'], lw=1.3))
    ax1.set_xlabel('Epoch', fontsize=FS_BODY, color=C['text_body'])
    ax1.set_ylabel('Loss',  fontsize=FS_BODY, color=C['text_body'])
    ax1.set_title('Training & Validation Loss', fontsize=FS_HEAD,
                  color=C['text_h1'], weight='bold', pad=10)
    ax1.legend(fontsize=FS_SMALL, loc='upper right', framealpha=0.85,
               edgecolor=C['border_soft'])

    # ── Accuracy chart (middle) ──
    ax2 = fig.add_axes([0.12, 0.18, 0.80, 0.32])
    _style(ax2)
    ax2.axvspan(1, 5, alpha=0.10, color=C['warmup_fill'], label='Warmup')
    ax2.plot(epochs, train_acc, 'o-', color=C['chart_1'], lw=2.5, ms=5, label='Train Acc', zorder=5)
    ax2.plot(epochs, val_acc, 's--', color=C['chart_2'], lw=2.0, ms=4, label='Val Acc', zorder=5)
    ax2.axvline(9, color=C['chart_mark'], ls=':', lw=1.8, alpha=0.85)
    ax2.annotate('Best R@1', xy=(9, train_acc[8]), xytext=(6.5, 30),
                 fontsize=FS_BODY, color=C['chart_mark'], weight='bold',
                 arrowprops=dict(arrowstyle='->', color=C['chart_mark'], lw=1.3))
    ax2.set_xlabel('Epoch',        fontsize=FS_BODY, color=C['text_body'])
    ax2.set_ylabel('Accuracy (%)', fontsize=FS_BODY, color=C['text_body'])
    ax2.set_title('Training & Validation Accuracy', fontsize=FS_HEAD,
                  color=C['text_h1'], weight='bold', pad=10)
    ax2.legend(fontsize=FS_SMALL, loc='lower right', framealpha=0.85,
               edgecolor=C['border_soft'])

    # ── Bottom pills ──
    pills = ["50 Epochs Total", "Best: Epoch 9", "5-Epoch LR Warmup",
             "Cosine Decay", "Batch 32 · Accum ×2"]
    pw, ph = 0.175, 0.035
    total_w = len(pills)*pw
    gap = (1.0 - total_w) / (len(pills)+1)
    for i, txt in enumerate(pills):
        px = gap + i*(pw+gap)
        ax_bg.add_patch(FancyBboxPatch(
            (px+0.003, 0.027), pw, ph,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor=C['chart_1'], edgecolor='none', alpha=0.06,
            zorder=4, transform=ax_bg.transAxes))
        ax_bg.add_patch(FancyBboxPatch(
            (px, 0.03), pw, ph,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor=C['card_1'], edgecolor=C['border'], linewidth=1.0,
            alpha=0.92, zorder=5, transform=ax_bg.transAxes))
        ax_bg.text(px+pw/2, 0.03+ph/2, txt, fontsize=FS_SMALL,
                   ha='center', va='center', color=C['text_h1'],
                   weight='bold', zorder=6, transform=ax_bg.transAxes)

    save(fig, "slide11_stage2_results.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating diagrams …\n")
    for i, (name, fn) in enumerate([
        ("Slide 5  — Data Pipeline",         generate_slide05),
        ("Slide 7  — Stage 1 GNN Training",  generate_slide07),
        ("Slide 8  — QFormer Bridge",        generate_slide08),
        ("Slide 11 — Stage 2 Results",       generate_slide11),
    ], 1):
        print(f"[{i}/4] {name}")
        fn()
    print(f"\nDone — all saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
