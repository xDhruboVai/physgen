"""Demo clip: one scene's c0 next to its c1, c2 and c3 versions, violation interval highlighted.
Usage: python make_demo.py <dataset_root> <scene_seed> <out_stem> [--still]"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from physgen import writer  # noqa: E402

NAMES = {
    "S1.1": "shadow points the wrong way", "S1.2": "shadow drifts off the object", "S1.3": "frozen shadow",
    "S1.4": "wrong shadow length", "S1.5": "shadow moves the wrong way", "S1.6": "wrong shadow blur",
    "S2.1": "objects pass through each other", "S2.2": "push with no contact", "S2.3": "gravity changes",
    "S2.4": "stops in mid-air", "S2.5": "bounces too high", "S2.6": "masses swapped",
    "S2.7": "spins with no torque", "S2.8": "friction changes", "S3.1": "target moves before it is hit",
    "S3.2": "domino falls too early", "S3.3": "stack topples too early", "S3.4": "slides before release",
}
CLASS = {0: "c0  physically correct", 1: "c1  shadow / lighting", 2: "c2  mechanics", 3: "c3  causality"}
RED, GREY, WHITE = (230, 40, 40), (90, 90, 90), (255, 255, 255)


def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load(root, seed):
    clips = pd.read_parquet(os.path.join(root, "manifest", "clips.parquet"))
    sel = clips[(clips["scene_seed"] == seed) & (clips["renderer"] == "raster")].sort_values("class")
    want = set(sel["clip_id"])
    frames = {}
    for key, sample in writer.read_shard(os.path.join(root, sel["shard"].iloc[0])):
        if key in want:
            frames[key] = writer.decode(sample)
    return [(r, frames[r["clip_id"]]) for _, r in sel.iterrows()]


def tile(frame, row, t, T, size):
    f1, f2 = font(int(size * 0.062), True), font(int(size * 0.052))
    head, bar = int(size * 0.2), int(size * 0.09)
    img = Image.new("RGB", (size, head + size + bar), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((8, 4), CLASS[int(row["class"])], font=f1, fill=WHITE)
    if row["class"] > 0:
        d.text((8, 6 + int(size * 0.075)), f"{row['subtype']}: {NAMES[row['subtype']]} ({row['severity']})", font=f2,
               fill=(220, 220, 220))
    img.paste(Image.fromarray(frame).resize((size, size), Image.BICUBIC), (0, head))
    on, off = (int(row["onset_frame"]), int(row["offset_frame"])) if row["class"] > 0 else (None, None)
    active = on is not None and on <= t < off
    if active:
        d.rectangle([0, head, size - 1, head + size - 1], outline=RED, width=5)
        d.text((size - int(size * 0.36), head + 8), "VIOLATION", font=f1, fill=RED)
    y0, y1 = head + size + bar // 3, head + size + 2 * bar // 3
    d.rectangle([8, y0, size - 8, y1], fill=GREY)
    x = lambda k: 8 + (size - 16) * k / T
    if on is not None:
        d.rectangle([x(on), y0, x(off), y1], fill=RED)
    d.rectangle([x(t) - 1, y0 - 4, x(t) + 1, y1 + 4], fill=WHITE)
    return img


def grid(clips, t, size):
    T = clips[0][1].shape[0]
    tiles = [tile(f[t], r, t, T, size) for r, f in clips]
    w, h = tiles[0].size
    out = Image.new("RGB", (2 * w + 6, 2 * h + 6 + 34), (12, 12, 12))
    for k, im in enumerate(tiles[:4]):
        out.paste(im, ((k % 2) * (w + 6), 34 + (k // 2) * (h + 6)))
    ImageDraw.Draw(out).text((8, 6), f"scene {clips[0][0]['scene_seed']} ({clips[0][0]['scenario'].replace('_', ' ')})   "
                                     f"frame {t + 1}/{T}   half speed   red = labelled violation",
                             font=font(max(12, size // 24)), fill=(200, 200, 200))
    return out


def main():
    root, seed, stem = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    clips = load(root, seed)
    assert [int(r["class"]) for r, _ in clips] == [0, 1, 2, 3], "scene needs c0-c3"
    T = clips[0][1].shape[0]
    if "--still" in sys.argv:  # one frame inside each violation, to choose a scene
        mids = [(int(r["onset_frame"]) + int(r["offset_frame"])) // 2 for r, _ in clips[1:]]
        Image.fromarray(np.concatenate([np.asarray(grid(clips, m, 260)) for m in mids], 1)).save(stem + "_still.png")
        return
    big = [grid(clips, t, 400) for t in range(T)]
    import imageio_ffmpeg
    W, H = big[0].size
    W, H = W - W % 2, H - H % 2
    w = imageio_ffmpeg.write_frames(stem + ".mp4", (W, H), fps=12, macro_block_size=1,
                                    output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    w.send(None)
    for im in big + big:  # plays twice
        w.send(np.ascontiguousarray(np.asarray(im)[:H, :W]))
    w.close()
    small = [grid(clips, t, 240).convert("P", palette=Image.ADAPTIVE, colors=128) for t in range(T)]
    small[0].save(stem + ".gif", save_all=True, append_images=small[1:], duration=83, loop=0, optimize=True)
    print("wrote", stem + ".mp4", stem + ".gif")


if __name__ == "__main__":
    main()
