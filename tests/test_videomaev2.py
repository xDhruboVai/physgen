"""The VideoMAE V2 weight mapping must reproduce the original architecture exactly.
Reference: an independent forward pass written from the published model definition (ViT with q/v
biases, fixed sinusoidal positions, mean pooling + fc_norm). Run: python tests/test_videomaev2.py"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physgen import features  # noqa: E402

MODEL = "OpenGVLab/VideoMAEv2-Base"


def sinusoid(n, d):
    pos = np.arange(n)[:, None] / np.power(10000, 2 * (np.arange(d) // 2) / d)[None]
    pos[:, 0::2], pos[:, 1::2] = np.sin(pos[:, 0::2]), np.cos(pos[:, 1::2])
    return torch.tensor(pos, dtype=torch.float32)[None]


def reference(src, x, heads=12, eps=1e-6):
    """x: (B, C, T, H, W) -> pooled features as in VideoMAEv2.forward_features."""
    h = F.conv3d(x, src["patch_embed.proj.weight"], src["patch_embed.proj.bias"], stride=(2, 16, 16)).flatten(2).transpose(1, 2)
    B, N, D = h.shape
    h = h + sinusoid(N, D)
    i = 0
    while f"blocks.{i}.norm1.weight" in src:
        g = lambda k: src[f"blocks.{i}.{k}"]
        y = F.layer_norm(h, (D,), g("norm1.weight"), g("norm1.bias"), eps)
        bias = torch.cat([g("attn.q_bias"), torch.zeros_like(g("attn.v_bias")), g("attn.v_bias")])
        qkv = F.linear(y, g("attn.qkv.weight"), bias).reshape(B, N, 3, heads, D // heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * (D // heads) ** -0.5, qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)).softmax(-1)
        h = h + F.linear((a @ v).transpose(1, 2).reshape(B, N, D), g("attn.proj.weight"), g("attn.proj.bias"))
        y = F.layer_norm(h, (D,), g("norm2.weight"), g("norm2.bias"), eps)
        h = h + F.linear(F.gelu(F.linear(y, g("mlp.fc1.weight"), g("mlp.fc1.bias"))), g("mlp.fc2.weight"), g("mlp.fc2.bias"))
        i += 1
    return F.layer_norm(h.mean(1), (D,), src["fc_norm.weight"], src["fc_norm.bias"], eps)


def test_videomaev2_mapping_matches_reference():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    model, post, _, tubelet = features.load_videomaev2(MODEL)
    src = {k.removeprefix("model."): v.float() for k, v in load_file(hf_hub_download(MODEL, "model.safetensors")).items()}
    x = torch.randn(2, 3, 16, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = reference(src, x)
        h = model.eval()(pixel_values=x.permute(0, 2, 1, 3, 4)).last_hidden_state
        ours = post(h.mean(1))
        slots = features.temporal_tokens(h, 16 // tubelet)
    err = (ours - ref).abs().max().item()
    assert err < 1e-3, f"max abs difference {err}"
    assert slots.shape == (2, 8, 768) and torch.allclose(slots.mean(1), h.mean(1), atol=1e-5)
    print("max abs difference", err)


if __name__ == "__main__":
    test_videomaev2_mapping_matches_reference()
    print("ok test_videomaev2_mapping_matches_reference")
