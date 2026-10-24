"""Optional precomputed frozen-backbone features (D6 option D).

Backbone-agnostic: any Hugging Face video model that transformers implements natively works
(VideoMAE, TimeSformer, V-JEPA 2, ...). Repository code is never executed (no trust_remote_code);
VideoMAE V2 checkpoints (the doc's backbone) ship only custom code, so their safetensors weights are
mapped onto the built-in VideoMAE architecture instead (same ViT, see load_videomaev2).

Features keep time: tokens are mean-pooled over space only, giving one vector per tubelet
(2 frames for VideoMAE) per window. Every export goes to features/<model>-w<window>-s<stride>/,
one .npz per shard: <clip_id> -> (n_windows, slots_per_window, dim) float16, plus index.parquet.
"""
import json
import os
import zipfile

import numpy as np
import pandas as pd

from . import writer
from .audit import _load, iter_clips

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "OpenGVLab/VideoMAEv2-Base"
# model downloads go to the project drive, not the user profile on C:. Set at import: huggingface_hub
# fixes its cache path when it is first imported.
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "third_party", "hf-cache"))


def _hf_home():
    import huggingface_hub.constants as c
    if not os.path.abspath(c.HF_HUB_CACHE).startswith(os.path.abspath(os.environ["HF_HOME"])):
        raise RuntimeError(f"huggingface_hub was imported before physgen.features; cache would be {c.HF_HUB_CACHE}")


def load_videomaev2(model_id):
    """VideoMAE V2 (ViT, q/v biases, fixed sinusoidal positions, mean pooling + fc_norm) as a built-in
    transformers VideoMAEModel. Returns (model, fc_norm, processor, tubelet)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEModel
    _hf_home()
    with open(hf_hub_download(model_id, "config.json")) as f:
        mc = json.load(f)["model_config"]
    with open(hf_hub_download(model_id, "preprocessor_config.json")) as f:
        pc = json.load(f)
    if mc.get("init_values", 0) or mc.get("cos_attn") or mc.get("use_learnable_pos_emb"):
        raise NotImplementedError(f"{model_id}: layer scale / cosine attention / learned positions not mapped")
    cfg = VideoMAEConfig(image_size=mc["img_size"], patch_size=mc["patch_size"], num_channels=mc["in_chans"],
                         num_frames=mc["num_frames"], tubelet_size=mc["tubelet_size"], hidden_size=mc["embed_dim"],
                         num_hidden_layers=mc["depth"], num_attention_heads=mc["num_heads"],
                         intermediate_size=int(mc["embed_dim"] * mc["mlp_ratio"]), qkv_bias=mc["qkv_bias"],
                         layer_norm_eps=mc["layer_norm_eps"], use_mean_pooling=True)
    src = load_file(hf_hub_download(model_id, "model.safetensors"))
    src = {k[len("model."):] if k.startswith("model.") else k: v for k, v in src.items()}
    D = mc["embed_dim"]
    sd = {"embeddings.patch_embeddings.projection.weight": src["patch_embed.proj.weight"],
          "embeddings.patch_embeddings.projection.bias": src["patch_embed.proj.bias"]}
    for i in range(mc["depth"]):
        b, o = f"blocks.{i}.", f"encoder.layer.{i}."
        q, k, v = src[b + "attn.qkv.weight"].split(D)
        sd.update({o + "attention.attention.query.weight": q, o + "attention.attention.key.weight": k,
                   o + "attention.attention.value.weight": v,
                   o + "attention.attention.query.bias": src[b + "attn.q_bias"],
                   o + "attention.attention.value.bias": src[b + "attn.v_bias"],
                   o + "attention.attention.key.bias": src[b + "attn.q_bias"] * 0,  # V2 has no key bias
                   o + "attention.output.dense.weight": src[b + "attn.proj.weight"],
                   o + "attention.output.dense.bias": src[b + "attn.proj.bias"],
                   o + "layernorm_before.weight": src[b + "norm1.weight"], o + "layernorm_before.bias": src[b + "norm1.bias"],
                   o + "layernorm_after.weight": src[b + "norm2.weight"], o + "layernorm_after.bias": src[b + "norm2.bias"],
                   o + "intermediate.dense.weight": src[b + "mlp.fc1.weight"], o + "intermediate.dense.bias": src[b + "mlp.fc1.bias"],
                   o + "output.dense.weight": src[b + "mlp.fc2.weight"], o + "output.dense.bias": src[b + "mlp.fc2.bias"]})
    model = VideoMAEModel(cfg)
    names = set(model.state_dict())
    if not any(n.endswith("attention.attention.key.bias") for n in names):
        sd = {k: v for k, v in sd.items() if not k.endswith("key.bias")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "position" not in m]  # fixed sinusoid table is a buffer, not a weight
    if missing or unexpected:
        raise RuntimeError(f"VideoMAE V2 weight mapping incomplete: missing {missing[:5]}, unexpected {unexpected[:5]}")
    fc_norm = torch.nn.LayerNorm(D, eps=mc["layer_norm_eps"])
    fc_norm.load_state_dict({"weight": src["fc_norm.weight"], "bias": src["fc_norm.bias"]})
    return model, fc_norm, TensorVideoProcessor(pc["size"], pc["image_mean"], pc["image_std"]), mc["tubelet_size"]


class TensorVideoProcessor:
    """Same steps as VideoMAEImageProcessor (shortest-edge resize, centre crop, normalise) on tensors;
    the PIL path costs more than the encoder on small frames."""

    def __init__(self, size, mean, std):
        import torch
        self.size = size
        self.mean, self.std = torch.tensor(mean).view(1, 1, 3, 1, 1), torch.tensor(std).view(1, 1, 3, 1, 1)

    def __call__(self, vids, return_tensors="pt"):
        import torch
        import torch.nn.functional as F
        x = torch.as_tensor(np.stack([np.stack(v) for v in vids])).permute(0, 1, 4, 2, 3).float() / 255  # B,T,C,H,W
        B, T, C, H, W = x.shape
        if min(H, W) != self.size:
            s = self.size / min(H, W)
            x = F.interpolate(x.reshape(B * T, C, H, W), size=(round(H * s), round(W * s)), mode="bilinear",
                              antialias=True, align_corners=False).reshape(B, T, C, round(H * s), round(W * s))
            H, W = x.shape[-2:]
        top, left = (H - self.size) // 2, (W - self.size) // 2
        x = x[..., top:top + self.size, left:left + self.size]
        return {"pixel_values": (x - self.mean) / self.std}


def load_backbone(model_id):
    """(model, post-norm or None, processor, tubelet or None) without executing repository code."""
    if "videomaev2" in model_id.lower():
        return load_videomaev2(model_id)
    _hf_home()
    import transformers
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id)
    proc = None
    for name in ("AutoVideoProcessor", "AutoImageProcessor"):
        cls = getattr(transformers, name, None)
        try:
            proc = cls.from_pretrained(model_id) if cls else None
        except Exception:
            proc = None
        if proc is not None:
            break
    if proc is None:
        raise RuntimeError(f"no video or image processor for {model_id}")
    return model, None, proc, getattr(model.config, "tubelet_size", None)


def temporal_tokens(hidden, slots):
    """(B, N, D) tokens -> (B, slots, D): mean over space within each time slot (time-major token order)."""
    B, N, D = hidden.shape
    if slots and N % slots == 0:
        return hidden.reshape(B, slots, N // slots, D).mean(2)
    return hidden.mean(1, keepdim=True)


def _scenes(clip_iter):
    """Group consecutive (row, sample) pairs by scene and decode them; clips of a scene share a shard."""
    group, key = [], None
    for row, sample in clip_iter:
        k = (row["shard"], row["scene_seed"])
        if group and k != key:
            yield group
            group = []
        key = k
        group.append((row, writer.decode(sample)))
    if group:
        yield group


def _prefetch(it, depth=4):
    """Run an iterator on a background thread so JPEG decoding overlaps with GPU work."""
    import queue
    import threading
    q, end = queue.Queue(depth), object()

    def run():
        try:
            for x in it:
                q.put(x)
            q.put(end)
        except BaseException as e:  # surface producer errors in the consumer
            q.put(e)

    threading.Thread(target=run, daemon=True).start()
    while True:
        x = q.get()
        if x is end:
            return
        if isinstance(x, BaseException):
            raise x
        yield x


def export(root, model_id=DEFAULT_MODEL, window=16, stride=8, batch=8, device="cuda"):
    import torch
    clips, _ = _load(root)
    model, post, proc, tubelet = load_backbone(model_id)
    half = device.startswith("cuda")
    model = model.to(device).eval()
    post = post.to(device).eval() if post is not None else None
    if half:
        model, post = model.half(), (post.half() if post is not None else None)
    slots = window // tubelet if tubelet else 0
    slug = f"{model_id.replace('/', '__')}-w{window}-s{stride}"
    out_dir = os.path.join(root, "features", slug)
    os.makedirs(out_dir, exist_ok=True)
    feats, shard = {}, None
    npz = lambda s: os.path.basename(s).replace(".tar", ".npz")
    starts_of = lambda T: list(range(0, max(T - window, 0) + 1, stride))
    entry = lambda cid, s, T, cls, split: dict(clip_id=cid, file=npz(s), window=window, stride=stride,
                                               window_starts=starts_of(T), **{"class": int(cls)}, split=split)

    def complete(s):  # resume: a shard whose feature file already holds all its clips is skipped
        try:
            with np.load(os.path.join(out_dir, npz(s))) as z:
                return set(clips.loc[clips["shard"] == s, "clip_id"]) <= set(z.files)
        except (OSError, ValueError, zipfile.BadZipFile):
            return False

    done = {s for s in clips["shard"].unique() if complete(s)}
    if done:
        print(f"resuming: {len(done)} shards already exported")
    index = [entry(r.clip_id, r.shard, r.num_frames, r["class"], r.split)
             for _, r in clips[clips["shard"].isin(done)].iterrows()]

    def flush():
        if shard is not None and feats:
            path = os.path.join(out_dir, npz(shard))
            with open(path + ".part", "wb") as f:  # atomic: an interrupted write never looks complete
                np.savez(f, **feats)
            os.replace(path + ".part", path)
            feats.clear()

    def encode(windows):
        inputs = proc(windows, return_tensors="pt")
        inputs = {k: (v.to(device).half() if v.is_floating_point() and half else v.to(device)) for k, v in inputs.items()}
        o = model(**inputs)
        h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        x = temporal_tokens(h, slots) if h.dim() == 3 else h[:, None]
        return (post(x) if post is not None else x).float().cpu().numpy()

    with torch.no_grad():
        for group in _prefetch(_scenes(iter_clips(root, clips[~clips["shard"].isin(done)]))):
            if group[0][0]["shard"] != shard:
                flush()
                shard = group[0][0]["shard"]
            # one batch stream per scene; windows ending before a violating clip's onset are copied from
            # the scene's c0 clip (pre-onset frames are identical)
            c0 = next((g for g, (row, _) in enumerate(group) if row["class"] == 0), None)
            src, jobs = {}, []
            for g, (row, frames) in enumerate(group):
                for k, s in enumerate(starts_of(len(frames))):
                    if c0 is not None and g != c0 and s + window <= int(row["onset_frame"]):
                        src[g, k] = (c0, k)
                    else:
                        src[g, k] = (g, k)
                        jobs.append((g, k, s))
            res = {}
            for b in range(0, len(jobs), batch):
                part = jobs[b:b + batch]
                res.update(zip([(g, k) for g, k, _ in part],
                               encode([list(group[g][1][s:s + window]) for g, _, s in part])))
            for g, (row, frames) in enumerate(group):
                n = len(starts_of(len(frames)))
                feats[row.name] = np.stack([res[src[g, k]] for k in range(n)]).astype(np.float16)
                index.append(entry(row.name, shard, len(frames), row["class"], row["split"]))
    flush()
    pd.DataFrame(index).to_parquet(os.path.join(out_dir, "index.parquet"))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(dict(model=model_id, window=window, stride=stride, tubelet=tubelet,
                       slots_per_window=slots or 1, frames_per_slot=(window // slots) if slots else window), f, indent=1)
    print(f"features for {len(index)} clips -> {out_dir}")
