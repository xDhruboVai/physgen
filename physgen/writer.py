"""Frames + metadata into WebDataset tar shards, plus parquet manifests (doc 5.5, D6).

A shard and its manifest part are committed together when the shard closes, so an interrupted
run never leaves manifest rows pointing at a truncated shard; `generate` resumes by skipping
scenes whose config was committed.
"""
import glob
import io
import json
import os
import tarfile
import time

import numpy as np

FORMATS = ("jpeg", "png", "npy", "mp4")


def to_json(x):
    return json.dumps(x, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist())


def encode(frames, cfg):
    """frames (T,H,W,3) uint8 -> {extension: bytes}. Both clips of a pair use identical settings."""
    fmt = cfg["output"]["format"]
    if fmt in ("jpeg", "png"):
        from PIL import Image
        out = {}
        for t, f in enumerate(frames):
            b = io.BytesIO()
            if fmt == "jpeg":
                Image.fromarray(f).save(b, "JPEG", quality=cfg["output"]["jpeg_quality"], subsampling=0)
                out[f"{t:03d}.jpg"] = b.getvalue()
            else:
                Image.fromarray(f).save(b, "PNG", compress_level=6)
                out[f"{t:03d}.png"] = b.getvalue()
        return out
    if fmt == "npy":
        b = io.BytesIO()
        np.save(b, frames)
        return {"frames.npy": b.getvalue()}
    if fmt == "mp4":
        import tempfile
        import imageio_ffmpeg
        T, H, W, _ = frames.shape
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            w = imageio_ffmpeg.write_frames(path, (W, H), fps=cfg["dataset"]["clip"]["fps"], codec="libx264",
                                            macro_block_size=1, output_params=["-crf", str(cfg["output"].get("crf", 18))])
            w.send(None)
            for f in frames:
                w.send(np.ascontiguousarray(f))
            w.close()
            with open(path, "rb") as f:
                return {"mp4": f.read()}
        finally:
            os.remove(path)
    raise ValueError(f"output.format must be one of {FORMATS}")


def decode(sample):
    """{extension: bytes} -> frames (T,H,W,3) uint8 (inverse of encode)."""
    if "frames.npy" in sample:
        return np.load(io.BytesIO(sample["frames.npy"]))
    if "mp4" in sample:
        import imageio.v3 as iio
        return iio.imread(sample["mp4"], extension=".mp4")
    from PIL import Image
    keys = sorted(k for k in sample if k.endswith((".jpg", ".png")))
    return np.stack([np.asarray(Image.open(io.BytesIO(sample[k])).convert("RGB")) for k in keys])


def read_shard(path):
    """Yield (clip_id, {extension: bytes}) in tar order (WebDataset convention: key = text before the first dot)."""
    cur, sample = None, {}
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, ext = m.name.split(".", 1)
            if key != cur and cur is not None:
                yield cur, sample
                sample = {}
            cur = key
            sample[ext] = tf.extractfile(m).read()
    if cur is not None:
        yield cur, sample


class ShardWriter:
    def __init__(self, root, split, max_clips):
        self.root, self.split, self.max_clips = root, split, max_clips
        self.run = time.strftime("%Y%m%d-%H%M%S")
        self.k, self.tar, self.rows, self.configs = 0, None, [], []
        for d in ("shards/" + split, "manifest/parts", "configs/" + split, "logs"):
            os.makedirs(os.path.join(root, d), exist_ok=True)

    def _open(self):
        self.name = f"shards/{self.split}/shard-{self.run}-{self.k:05d}.tar"
        self.tar = tarfile.open(os.path.join(self.root, self.name + ".part"), "w")
        self.n = 0

    def _add(self, name, data):
        ti = tarfile.TarInfo(name)
        ti.size, ti.mtime = len(data), 0
        self.tar.addfile(ti, io.BytesIO(data))

    def write_scene(self, scene, clips):
        """clips: [(meta, {ext: bytes})]; all clips of a scene go to the same shard."""
        if self.tar is None:
            self._open()
        for meta, enc in clips:
            for ext, data in enc.items():
                self._add(f"{meta['clip_id']}.{ext}", data)
            self._add(f"{meta['clip_id']}.json", to_json(meta).encode())
            self.rows.append(dict(meta, shard=self.name))
            self.n += 1
        self.configs.append(scene)
        if self.n >= self.max_clips:
            self.close()

    def close(self):
        if self.tar is None:
            return
        self.tar.close()
        os.replace(os.path.join(self.root, self.name + ".part"), os.path.join(self.root, self.name))
        part = os.path.join(self.root, "manifest/parts", f"{self.split}-{self.run}-{self.k:05d}.parquet")
        rows_to_frame(self.rows).to_parquet(part)
        for sc in self.configs:  # committed last: a config on disk means its clips are in a closed shard
            with open(os.path.join(self.root, "configs", self.split, f"s{sc['scene_seed']:09d}.json"), "w") as f:
                f.write(to_json(sc))
        self.tar, self.rows, self.configs = None, [], []
        self.k += 1


NESTED = ("objects", "environment", "light", "camera", "violation_params")


def rows_to_frame(rows):
    import pandas as pd
    out = []
    for r in rows:
        r = dict(r)
        for k in NESTED:
            r[k] = to_json(r.get(k))
        r["resolution"] = to_json(r["resolution"])
        r["involved_objects"] = to_json(r.get("involved_objects", []))
        out.append(r)
    return pd.DataFrame(out)


def rebuild_manifests(root):
    """manifest/clips.parquet (one row per clip) and manifest/pairs.parquet (one row per pair)."""
    import pandas as pd
    parts = sorted(glob.glob(os.path.join(root, "manifest/parts/*.parquet")))
    if not parts:
        return None, None
    clips = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    clips = clips.drop_duplicates("clip_id", keep="last")
    clips.to_parquet(os.path.join(root, "manifest/clips.parquet"))
    c0 = set(clips.loc[clips["class"] == 0, "clip_id"])
    v = clips[clips["class"] > 0]
    c0_of = [p[:p.rfind("_c")] for p in v["pair_id"]]  # "<base>_c0_c<k>" -> "<base>_c0"
    pairs = pd.DataFrame(dict(pair_id=v["pair_id"].values, scene_seed=v["scene_seed"].values, split=v["split"].values,
                              c0_clip=[c if c in c0 else None for c in c0_of],
                              violating_clip=v["clip_id"].values, **{"class": v["class"].values},
                              subtype=v["subtype"].values, severity=v["severity"].values, timing=v["timing"].values,
                              onset_frame=v["onset_frame"].values, offset_frame=v["offset_frame"].values,
                              scenario=v["scenario"].values, shard=v["shard"].values))
    pairs.to_parquet(os.path.join(root, "manifest/pairs.parquet"))
    return clips, pairs


def committed_seeds(root, split):
    return {int(os.path.basename(p)[1:10]) for p in glob.glob(os.path.join(root, "configs", split, "s*.json"))}
