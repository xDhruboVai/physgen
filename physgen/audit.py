"""Dataset-level validation (V5, V8, V9, V10 on the written shards) and the shortcut audit (doc 9.2-9.3)."""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from . import validators, writer


def _load(root):
    clips, pairs = writer.rebuild_manifests(root)
    if clips is None:
        raise SystemExit(f"no manifests under {root}")
    return clips, pairs


def iter_clips(root, clips):
    """Yield (meta row, {ext: bytes}) for manifest rows, shard by shard."""
    want = set(clips["clip_id"])
    rows = clips.set_index("clip_id")
    for shard in sorted(clips["shard"].unique()):
        for key, sample in writer.read_shard(os.path.join(root, shard)):
            if key in want:
                yield rows.loc[key], sample


def validate(root):
    clips, pairs = _load(root)
    errors, warnings = [], []
    frames = {}
    seen = set()
    for row, sample in iter_clips(root, clips):
        meta = json.loads(sample["json"])
        seen.add(meta["clip_id"])
        try:
            validators.v8_labels(meta)
        except AssertionError as e:
            errors.append(f"{meta['clip_id']}: {e}")
        images = sorted(k for k in sample if k.endswith((".jpg", ".png")))
        # keep one digest per frame, not the frames (a pilot's frames do not fit in RAM); identical frames
        # encode to identical bytes, so equal digests before the onset is the V5 check on stored data
        raw = [sample[k] for k in images] if images else [f.tobytes() for f in writer.decode(sample)]
        frames[meta["clip_id"]] = [hashlib.md5(b).digest() for b in raw]
        n = len(frames[meta["clip_id"]])
        if n != meta["num_frames"]:
            errors.append(f"{meta['clip_id']}: {n} frames, expected {meta['num_frames']}")
    for cid in set(clips["clip_id"]) - seen:
        errors.append(f"{cid}: listed in the manifest but missing from its shard")
    # V5 on stored data: identical frames before onset (encoded bytes are identical for identical frames)
    for _, p in pairs.iterrows():
        a, b = frames.get(p["c0_clip"]), frames.get(p["violating_clip"])
        if a is None or b is None:
            errors.append(f"{p['pair_id']}: missing clip")
            continue
        on = int(p["onset_frame"])
        same = a[:on] == b[:on]
        if not same:
            errors.append(f"{p['pair_id']}: frames differ before onset {on} (V5)")
    # V9: near-duplicate scenes
    hashes = {}
    for path in glob.glob(os.path.join(root, "configs", "*", "*.json")):
        with open(path) as f:
            sc = json.load(f)
        h = validators.scene_hash(sc)
        if h in hashes and hashes[h] != sc["scene_seed"]:  # same seed in test_id and test_renderer is intended
            warnings.append(f"V9 near-duplicate scenes {hashes[h]} and {sc['scene_seed']}")
        hashes[h] = sc["scene_seed"]
    # orphan shards (from an interrupted run) carry no manifest rows
    listed = set(clips["shard"])
    for path in glob.glob(os.path.join(root, "shards", "*", "*.tar*")):
        rel = os.path.relpath(path, root).replace("\\", "/")
        if rel not in listed:
            warnings.append(f"orphan shard (not in any manifest): {rel}")
    # V10: balance report
    counts = clips.groupby(["split", "class"]).size().unstack(fill_value=0)
    print("clips per split and class:\n" + counts.to_string())
    print("pairs per subtype:\n" + pairs.groupby(["class", "subtype"]).size().to_string())
    for w in warnings:
        print("warning:", w)
    for e in errors[:50]:
        print("ERROR:", e)
    print(f"validate: {len(clips)} clips, {len(pairs)} pairs, {len(errors)} errors, {len(warnings)} warnings")
    return not errors


# ---------------------------------------------------------------- balance and probes
def _objects_table(clips):
    rows = []
    for _, c in clips.iterrows():
        for o in json.loads(c["objects"]):
            if o["role"] == "dynamic":
                rows.append(dict(split=c["split"], cls=c["class"], shape=o["shape"], material=o["material"],
                                 texture=o["texture"]))
    return pd.DataFrame(rows)


def balance(root, clips):
    out = os.path.join(root, "stats")
    os.makedirs(out, exist_ok=True)
    env = clips["environment"].map(json.loads)
    clips = clips.assign(floor_texture=env.map(lambda e: e["floor_texture"]), hdri=env.map(lambda e: e["hdri"]),
                         subtype=clips["subtype"].fillna("none"))
    tables = {}
    for col in ("subtype", "scenario", "floor_texture", "hdri", "severity", "timing"):
        tables[col] = clips.fillna({"severity": "none", "timing": "none"}).groupby(["split", col, "class"]).size().unstack(fill_value=0)
    objs = _objects_table(clips)
    for col in ("shape", "material", "texture"):
        tables[col] = objs.groupby(["split", col, "cls"]).size().unstack(fill_value=0)
    md = ["# Balance report", ""]
    for name, t in tables.items():
        t.to_csv(os.path.join(out, f"balance_{name}.csv"))
        if name in ("subtype", "scenario", "shape", "material", "severity", "timing"):
            md += [f"## {name}", "", t.to_markdown() if hasattr(t, "to_markdown") else t.to_string(), ""]
    with open(os.path.join(out, "balance.md"), "w") as f:
        f.write("\n".join(md))
    print(f"balance tables -> {out}")


def _features(root, clips, max_clips, rng):
    """Per-clip probe inputs, computed from the stored frames (what a model would see)."""
    import cv2
    keep = clips
    if len(clips) > max_clips:
        seeds = clips["scene_seed"].unique()
        seeds = rng.choice(seeds, size=max(1, int(len(seeds) * max_clips / len(clips))), replace=False)
        keep = clips[clips["scene_seed"].isin(seeds)]
    F = {}
    for row, sample in iter_clips(root, keep):
        x = writer.decode(sample).astype(np.float32) / 255.0
        T = len(x)
        small = np.stack([cv2.resize(f, (16, 16), interpolation=cv2.INTER_AREA) for f in x])
        gray = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA) for f in x])
        diff = np.abs(np.diff(gray, axis=0)).mean((1, 2))
        flow = []
        for t in range(0, T - 1, 2):
            fl = cv2.calcOpticalFlowFarneback((gray[t] * 255).astype(np.uint8), (gray[t + 1] * 255).astype(np.uint8),
                                              None, 0.5, 2, 9, 2, 5, 1.1, 0)
            mag = np.linalg.norm(fl, axis=-1)
            flow += [mag.mean(), mag.std(), np.percentile(mag, 95)]
        lap = [cv2.Laplacian(g, cv2.CV_32F).var() for g in gray]
        sizes = [len(v) for k, v in sorted(sample.items()) if k.endswith((".jpg", ".png"))] or [0] * T
        F[row.name] = dict(single=small[T // 2].ravel(), early=small[:8].mean(0).ravel(),
                           motion=np.r_[diff, flow], hf=np.r_[lap, np.asarray(sizes, float) / 1e4])
    return keep[keep["clip_id"].isin(F)], F


def _probe(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    k = min(5, len(np.unique(groups)))
    if k < 2 or len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(k))
    return float(balanced_accuracy_score(y, pred))


def run_probes(root, clips, max_clips=4000, seed=0):
    """P1-P6: baselines on information that should not solve the task; all should stay near chance.
    Balanced accuracy (chance = 1/classes) for c0-vs-violating and for the 4-class task, grouped by scene."""
    rng = np.random.default_rng(seed)
    keep, F = _features(root, clips, max_clips, rng)
    ids = keep["clip_id"].tolist()
    y4 = keep["class"].to_numpy()
    yb = (y4 > 0).astype(int)
    g = keep["scene_seed"].to_numpy()
    meta = pd.get_dummies(pd.DataFrame(dict(
        scenario=keep["scenario"],
        floor=keep["environment"].map(lambda e: json.loads(e)["floor_texture"]),
        hdri=keep["environment"].map(lambda e: json.loads(e)["hdri"]),
        shapes=keep["objects"].map(lambda o: "+".join(sorted(x["shape"] for x in json.loads(o) if x["role"] == "dynamic"))),
        materials=keep["objects"].map(lambda o: "+".join(sorted(x["material"] for x in json.loads(o) if x["role"] == "dynamic"))),
    ))).to_numpy(float)
    rows = []
    for pid, name, X in (("P1", "single frame", np.stack([F[i]["single"] for i in ids])),
                         ("P2", "first 8 frames (pre-onset)", np.stack([F[i]["early"] for i in ids])),
                         ("P3", "motion energy / optical flow", np.stack([F[i]["motion"] for i in ids])),
                         ("P4", "metadata (scenario, shapes, materials, floor, hdri)", meta),
                         ("P6", "high-frequency / compression stats", np.stack([F[i]["hf"] for i in ids]))):
        rows.append(dict(probe=pid, input=name, binary_bal_acc=_probe(X, yb, g), four_class_bal_acc=_probe(X, y4, g)))
    v = keep[keep["class"] > 0]
    T = v["num_frames"].to_numpy(float)
    for pid, name, X in (("P5", "onset position only (violating clips, 3-class)", (v["onset_frame"] / T).to_numpy()[:, None]),
                         ("P5b", "interval duration (informational: c3 is short by definition)",
                          ((v["offset_frame"] - v["onset_frame"]) / T).to_numpy()[:, None])):
        rows.append(dict(probe=pid, input=name, binary_bal_acc=float("nan"),
                         four_class_bal_acc=_probe(X, v["class"].to_numpy(), v["scene_seed"].to_numpy())))
    res = pd.DataFrame(rows)
    res["chance_binary"], res["chance_classes"] = 0.5, [1 / len(np.unique(y4))] * 5 + [1 / 3] * 2
    out = os.path.join(root, "stats", "probes.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print("Note: P1 above chance can be legitimate for c1 (a single frame can show a wrong shadow); report it.")
    return res


def audit(root, probes=True):
    clips, _ = _load(root)
    balance(root, clips)
    if probes:
        run_probes(root, clips)
