"""Clip timelines from exported backbone features (doc section 10: frozen encoder, sliding windows)."""
import json
import os

import numpy as np
import pandas as pd


def load(root, features, splits=None):
    """Returns (X float16 (N, slots, D), Y slot classes (N, slots), meta DataFrame, frames_per_slot).
    Overlapping windows are averaged per time slot, so every clip becomes one sequence of slots."""
    fdir = os.path.join(root, "features", features)
    with open(os.path.join(fdir, "info.json")) as f:
        info = json.load(f)
    index = pd.read_parquet(os.path.join(fdir, "index.parquet"))
    clips = pd.read_parquet(os.path.join(root, "manifest", "clips.parquet")).set_index("clip_id")
    if splits:
        index = index[index["split"].isin(splits)]
    fps_slot = info["frames_per_slot"]
    X, Y, rows = [], [], []
    for file, grp in index.groupby("file"):
        data = np.load(os.path.join(fdir, file))
        for _, r in grp.iterrows():
            w = data[r["clip_id"]].astype(np.float32)  # (n_windows, slots_per_window, D)
            meta = clips.loc[r["clip_id"]]
            T = int(meta["num_frames"])
            n = T // fps_slot
            acc, cnt = np.zeros((n, w.shape[-1]), np.float32), np.zeros(n)
            for k, s in enumerate(r["window_starts"]):
                a = s // fps_slot
                acc[a:a + w.shape[1]] += w[k]
                cnt[a:a + w.shape[1]] += 1
            X.append((acc / np.maximum(cnt, 1)[:, None]).astype(np.float16))
            fc = np.asarray(meta["frame_classes"]).reshape(n, fps_slot)
            Y.append(fc.max(1))  # a slot is violating if any of its frames is
            rows.append(dict(clip_id=r["clip_id"], **{c: meta[c] for c in (
                "pair_id", "scene_seed", "class", "subtype", "severity", "timing", "scenario", "split",
                "onset_frame", "offset_frame", "num_frames", "renderer")},
                camera_mode=json.loads(meta["camera"]).get("mode", "static")))
    meta = pd.DataFrame(rows)
    return np.stack(X), np.stack(Y), meta, fps_slot


