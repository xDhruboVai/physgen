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

import numpy as np
import pandas as pd

from . import writer
from .audit import _load, iter_clips

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "OpenGVLab/VideoMAEv2-Base"


def _hf_home():
    # model downloads go to the project drive, not the user profile on C:
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "third_party", "hf-cache"))


def load_videomaev2(model_id):
    """VideoMAE V2 (ViT, q/v biases, fixed sinusoidal positions, mean pooling + fc_norm) as a built-in
    transformers VideoMAEModel. Returns (model, fc_norm, processor, tubelet)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel
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
    proc = VideoMAEImageProcessor(do_resize=True, size={"shortest_edge": pc["size"]}, do_center_crop=True,
                                  crop_size={"height": pc["size"], "width": pc["size"]},
                                  image_mean=pc["image_mean"], image_std=pc["image_std"])
    return model, fc_norm, proc, mc["tubelet_size"]


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
    index, feats, shard = [], {}, None

    def flush():
        if shard is not None and feats:
            np.savez(os.path.join(out_dir, os.path.basename(shard).replace(".tar", ".npz")), **feats)
            feats.clear()

    with torch.no_grad():
        for row, sample in iter_clips(root, clips):
            if row["shard"] != shard:
                flush()
                shard = row["shard"]
            frames = writer.decode(sample)
            T = len(frames)
            starts = list(range(0, max(T - window, 0) + 1, stride))
            out = []
            for b in range(0, len(starts), batch):
                vids = [list(frames[s:s + window]) for s in starts[b:b + batch]]
                inputs = proc(vids, return_tensors="pt")  # first positional argument is the batch of videos
                inputs = {k: (v.to(device).half() if v.is_floating_point() and half else v.to(device)) for k, v in inputs.items()}
                o = model(**inputs)
                h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
                x = temporal_tokens(h, slots) if h.dim() == 3 else h[:, None]
                if post is not None:
                    x = post(x)
                out.append(x.float().cpu().numpy())
            feats[row.name] = np.concatenate(out).astype(np.float16)
            lab = np.asarray(row["frame_labels"])
            index.append(dict(clip_id=row.name, file=os.path.basename(shard).replace(".tar", ".npz"), window=window,
                              stride=stride, window_starts=starts, **{"class": int(row["class"])}, split=row["split"]))
    flush()
    pd.DataFrame(index).to_parquet(os.path.join(out_dir, "index.parquet"))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(dict(model=model_id, window=window, stride=stride, tubelet=tubelet,
                       slots_per_window=slots or 1, frames_per_slot=(window // slots) if slots else window), f, indent=1)
    print(f"features for {len(index)} clips -> {out_dir}")
