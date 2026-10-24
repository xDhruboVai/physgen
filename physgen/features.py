"""Optional precomputed frozen-backbone features (D6 option D).

Backbone-agnostic: any Hugging Face video model id with a video/image processor works
(VideoMAE, VideoMAEv2, V-JEPA 2, TimeSformer, ...). Features depend on the window, stride and
preprocessing, so every export goes to its own folder: features/<model>-w<window>-s<stride>/.
One .npz per shard: <clip_id> -> (n_windows, dim) float16, plus a parquet index with per-window labels.
"""
import os

import numpy as np
import pandas as pd

from . import writer
from .audit import _load, iter_clips


def _processor(model_id):
    import transformers
    for name in ("AutoVideoProcessor", "AutoImageProcessor"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            continue
    raise RuntimeError(f"no video or image processor found for {model_id}")


def _pool(out):
    """Mean-pooled token features from whatever the backbone returns."""
    import torch
    if isinstance(out, torch.Tensor):
        x = out
    elif getattr(out, "last_hidden_state", None) is not None:
        x = out.last_hidden_state
    elif getattr(out, "pooler_output", None) is not None:
        x = out.pooler_output
    else:
        x = out[0]
    return (x.mean(1) if x.dim() == 3 else x.reshape(len(x), -1)).float().cpu().numpy()


def export(root, model_id, window=16, stride=8, batch=8, device="cuda"):
    # model downloads go to the project drive, not the user profile on C:
    os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "third_party", "hf-cache"))
    import torch
    from transformers import AutoModel
    clips, _ = _load(root)
    proc = _processor(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    if device.startswith("cuda"):
        model = model.half()
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
                inputs = {k: (v.to(device).half() if v.is_floating_point() and device.startswith("cuda") else v.to(device))
                          for k, v in inputs.items()}
                out.append(_pool(model(**inputs)))
            feats[row.name] = np.concatenate(out).astype(np.float16)
            lab = np.asarray(row["frame_labels"])
            index.append(dict(clip_id=row.name, file=os.path.basename(shard).replace(".tar", ".npz"),
                              window_starts=starts, window_label=[int(lab[s:s + window].max()) for s in starts],
                              window_label_frac=[float(lab[s:s + window].mean()) for s in starts],
                              **{"class": int(row["class"])}, split=row["split"]))
    flush()
    pd.DataFrame(index).to_parquet(os.path.join(out_dir, "index.parquet"))
    print(f"features for {len(index)} clips -> {out_dir}")
