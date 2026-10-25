"""Architecture diagram: scene seed -> physics -> violation injection -> render -> labels -> baseline.
Usage: python make_diagram.py <out_stem>   (writes .png, .svg, .pdf)"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

GEN, INJ, OUT, MODEL = "#dbe9f6", "#fde2cf", "#e2f0d9", "#ece2f4"
EDGE = "#3a3a3a"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=11.5, weight="bold", color="#111")
    ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=8.6, color="#222", linespacing=1.35)


def arrow(ax, a, b, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=EDGE,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", fontsize=8, style="italic", color="#444")


def main(stem):
    fig, ax = plt.subplots(figsize=(16, 8.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.text(0.2, 8.35, "physgen: synthetic physics-violation dataset and baseline", fontsize=15, weight="bold")
    ax.text(0.2, 7.98, "Every clip is a pure function of (scene seed, generator commit, asset manifest). "
                       "Both clips of a pair share everything except the injected violation.", fontsize=9.5, color="#333")

    w, h, y1 = 2.45, 2.2, 5.15
    xs = [0.2, 2.85, 5.5, 8.15, 10.8, 13.45]
    box(ax, xs[0], y1, w, h, "1  Scene seed",
        "split + index -> seed\nindependent RNG stream\nper module\nSobol draws for light\nand onsets", GEN)
    box(ax, xs[1], y1, w, h, "2  Scene sampler",
        "scenario (7 families)\nobjects: primitives, YCB\nmaterial -> mass, friction,\nrestitution (look sampled\nindependently)\nfloor, HDRI, key light", GEN)
    box(ax, xs[2], y1, w, h, "3  Physics (PyBullet)",
        "240 Hz recording,\n960 Hz solver\nc0 poses + contact events\ncamera framed on c0\nstatic distractors added\nchecks V1-V4, V7", GEN)
    box(ax, xs[3], y1, w, h, "4  Violation injection",
        "c1: shadow light / shadow\n    casters edited, c0 motion\nc2: simulation edited at\n    onset, then continues\nc3: effect time-advanced\n    by 4-14 frames", INJ)
    box(ax, xs[4], y1, w, h, "5  Renderer",
        "GPU rasterizer (moderngl):\nshading light + shadow light,\nPCSS shadows, HDRI ambient,\nmotion blur\nor Blender Cycles\n(renderer-transfer split)", GEN)
    box(ax, xs[5], y1, w, h, "6  Labels + checks",
        "frames before the change\ncopied from c0\nonset = first differing frame\noffset from the injector\nV5 identity, V6 visibility,\nV8 label consistency", OUT)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w, y1 + h / 2), (b, y1 + h / 2))

    # bottom row, left to right: dataset -> encoder -> head -> evaluation; audit reads the dataset
    w2, h2, y2 = 2.75, 2.0, 2.0
    bx = [0.2, 3.4, 6.6, 9.8, 13.0]
    box(ax, bx[0], y2, w2, h2, "7  Dataset",
        "WebDataset tar shards\n(JPEG frames + JSON labels)\nclips / pairs parquet\nscene configs; 8 splits\nincl. OOD + Blender", OUT)
    box(ax, bx[1], y2, w2, h2, "8  Frozen encoder",
        "VideoMAE V2-Base (ViT-B)\n16-frame windows\nspace-pooled tokens:\none feature per 2 frames", MODEL)
    box(ax, bx[2], y2, w2, h2, "9  Temporal head",
        "2-layer transformer over\nthe clip timeline\nper step: normal, c1, c2, c3\noptional pairwise loss", MODEL)
    box(ax, bx[3], y2, w2, h2, "10  Evaluation",
        "AUROC, pair ranking,\n4-class macro F1\nonset / offset error,\ntIoU, segment F1\nper split and subtype", MODEL)
    box(ax, bx[4], y2, w2, h2, "Audit",
        "validate: V5, V8, V9, V10\nshortcut probes P1-P6\n(single frame, motion,\nmetadata, onset, codec)\nbalance tables", OUT)
    mid = y2 + h2 / 2
    arrow(ax, (bx[0] + w2, mid), (bx[1], mid))
    arrow(ax, (bx[1] + w2, mid), (bx[2], mid))
    arrow(ax, (bx[2] + w2, mid), (bx[3], mid))

    def elbow(points, text=None, dashed=False):
        xs_, ys_ = zip(*points)
        ax.plot(xs_[:-1], ys_[:-1], color=EDGE, lw=1.4, ls="--" if dashed else "-")
        arrow(ax, points[-2], points[-1])
        if text:
            ax.text((points[1][0] + points[2][0]) / 2, points[1][1] + 0.08, text, ha="center", fontsize=8,
                    style="italic", color="#444")

    # labels + checks (top right) down into the dataset (bottom left), routed through the gap between rows
    top_x = xs[5] + w / 2
    elbow([(top_x, y1), (top_x, 4.6), (bx[0] + w2 / 2, 4.6), (bx[0] + w2 / 2, y2 + h2)], "write clips + labels")
    elbow([(bx[0] + 0.9, y2), (bx[0] + 0.9, 1.55), (bx[3] + w2 / 2, 1.55), (bx[3] + w2 / 2, y2)], "frame labels")
    elbow([(bx[0] + 1.8, y2), (bx[0] + 1.8, 1.25), (bx[4] + w2 / 2, 1.25), (bx[4] + w2 / 2, y2)], "all clips and metadata")

    for color, label, x in ((GEN, "simulation and rendering", 0.2), (INJ, "violation injection", 3.6),
                            (OUT, "labels, data, audit", 6.4), (MODEL, "baseline model", 9.3)):
        ax.add_patch(FancyBboxPatch((x, 0.35), 0.35, 0.3, boxstyle="round,pad=0.01", fc=color, ec=EDGE))
        ax.text(x + 0.5, 0.5, label, va="center", fontsize=9.5)
    fig.savefig(stem + ".png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".svg", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", stem + ".png/.svg/.pdf")


if __name__ == "__main__":
    main(sys.argv[1])
