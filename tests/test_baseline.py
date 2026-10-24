"""Baseline sanity check on synthetic features that encode the true timeline: the head must learn it.
Run: python tests/test_baseline.py"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from baseline import metrics, train  # noqa: E402


def fake_dataset(root, scenes=80, T=64, D=16, seed=0):
    rng = np.random.default_rng(seed)
    feat = os.path.join(root, "features", "fake")
    os.makedirs(feat)
    os.makedirs(os.path.join(root, "manifest"))
    rows, index, arrays = [], [], {}
    starts = list(range(0, T - 16 + 1, 8))
    for split_i, split in enumerate(("train", "val", "test_id")):
        for s in range(scenes):
            seed_ = split_i * 10 ** 6 + s
            for cls in range(4):
                cid = f"s{seed_:09d}_c{cls}"
                fc = np.zeros(T, int)
                on = off = None
                if cls:
                    on = int(rng.integers(13, 48))
                    off = int(min(T, on + rng.integers(4, 20)))
                    fc[on:off] = cls
                slot_cls = fc.reshape(-1, 2).max(1)
                slot = rng.normal(0, 0.3, (T // 2, D))
                slot[np.arange(T // 2), slot_cls] += 2.0
                arrays[cid] = np.stack([slot[a // 2:a // 2 + 8] for a in starts]).astype(np.float16)
                rows.append(dict(clip_id=cid, pair_id=None if cls == 0 else f"s{seed_:09d}_c0_c{cls}", scene_seed=seed_,
                                 **{"class": cls}, subtype=None if cls == 0 else f"S{cls}.1", severity="clear",
                                 timing="persistent", scenario="free_fall", split=split, onset_frame=on, offset_frame=off,
                                 num_frames=T, renderer="raster", frame_classes=fc.tolist(), camera=json.dumps({"mode": "static"})))
                index.append(dict(clip_id=cid, file="all.npz", window_starts=starts, split=split, **{"class": cls}))
    np.savez(os.path.join(feat, "all.npz"), **arrays)
    pd.DataFrame(index).to_parquet(os.path.join(feat, "index.parquet"))
    pd.DataFrame(rows).to_parquet(os.path.join(root, "manifest", "clips.parquet"))
    with open(os.path.join(feat, "info.json"), "w") as f:
        json.dump(dict(frames_per_slot=2), f)


def test_head_learns_synthetic_timeline():
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "third_party", "tmp")) as root:
        fake_dataset(root)
        out = os.path.join(root, "run")
        train.train(root, out, features="fake", epochs=25, pairwise=True)
        m = train.evaluate(root, out)["test_id"]["summary"]
        assert m["auroc"] > 0.95 and m["pairwise_ranking_acc"] > 0.95, m
        assert m["segment_f1@0.5"] > 0.8 and m["onset_mae_frames"] < 3 and m["macro_f1_4class"] > 0.9, m


def test_best_segment():
    p = np.array([0, .9, .9, 0, .6, .6, .6, .6, 0])
    assert metrics.best_segment(p, 0.5) == (4, 8)
    assert metrics.best_segment(np.zeros(5), 0.5) is None


if __name__ == "__main__":
    test_best_segment()
    test_head_learns_synthetic_timeline()
    print("ok baseline tests")
