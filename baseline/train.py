"""Train the temporal head on frozen features and evaluate it on every split (doc section 10).

python -m baseline train    --root datasets/pilot --out results/head [--pairwise]
python -m baseline evaluate --root datasets/pilot --run results/head
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import data, metrics
from .model import TemporalHead, clip_scores

FEATURES = "OpenGVLab__VideoMAEv2-Base-w16-s8"


def _batches(meta, rng, scenes_per_batch):
    """Batches of whole scenes, so each c0 clip sits next to its violating partners (pairwise loss)."""
    seeds = rng.permutation(meta["scene_seed"].unique())
    groups = meta.groupby("scene_seed").indices
    for k in range(0, len(seeds), scenes_per_batch):
        yield np.concatenate([groups[s] for s in seeds[k:k + scenes_per_batch]])


def _pair_index(meta):
    """(violating row, c0 row) index pairs within meta."""
    pos = {cid: i for i, cid in enumerate(meta["clip_id"])}
    return [(i, pos[p[:p.rfind("_c")]]) for i, p in enumerate(meta["pair_id"]) if isinstance(p, str) and p[:p.rfind("_c")] in pos]


@torch.no_grad()
def infer(model, X, device, bs=256):
    model.eval()
    score, viol, cls = [], [], []
    for k in range(0, len(X), bs):
        lg = model(torch.as_tensor(X[k:k + bs], device=device).float())
        s, v = clip_scores(lg)
        score.append(s.cpu()), viol.append(v.cpu()), cls.append(lg[..., 1:].softmax(-1).cpu())
    return torch.cat(score).numpy(), torch.cat(viol).numpy(), torch.cat(cls).numpy()


def _val_metric(m):
    return 0.5 * m.get("auroc", 0) + 0.5 * m.get("segment_f1@0.5", 0)


def train(root, out, features=FEATURES, pairwise=False, epochs=60, lr=3e-4, scenes_per_batch=24, margin=1.0,
          seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, Y, M, fps_slot = data.load(root, features, ["train", "val"])
    tr, va = (M["split"] == "train").to_numpy(), (M["split"] == "val").to_numpy()
    Mt, Mv = M[tr].reset_index(drop=True), M[va].reset_index(drop=True)
    Xt, Yt, Xv = X[tr], Y[tr], X[va]
    model = TemporalHead(X.shape[-1], X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    freq = np.bincount(Yt.ravel(), minlength=4) + 1
    weight = torch.tensor((freq.sum() / (4 * freq)) ** 0.5, dtype=torch.float32, device=device)  # soft class balance
    pairs_t = _pair_index(Mt)
    best, history = (-1, None, None), []
    for ep in range(epochs):
        model.train()
        losses = []
        for idx in _batches(Mt, rng, scenes_per_batch):
            x = torch.as_tensor(Xt[idx], device=device).float()
            y = torch.as_tensor(Yt[idx], device=device).long()
            lg = model(x)
            loss = F.cross_entropy(lg.reshape(-1, 4), y.reshape(-1), weight=weight)
            if pairwise:  # violating clip must score above its c0 partner (doc 10 ablation)
                local = {g: k for k, g in enumerate(idx)}
                pp = [(local[a], local[b]) for a, b in pairs_t if a in local and b in local]
                if pp:
                    s, _ = clip_scores(lg)
                    a, b = torch.tensor(pp, device=device).T
                    loss = loss + F.relu(margin - (s[a] - s[b])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        sv, vv, cv = infer(model, Xv, device)
        m = metrics.summarize(metrics.predictions(Mv, sv, vv, cv, fps_slot, 0.0, 0.5))
        history.append(dict(epoch=ep, loss=float(np.mean(losses)), **{k: m.get(k) for k in ("auroc", "segment_f1@0.5", "mean_tiou")}))
        if _val_metric(m) > best[0]:
            best = (_val_metric(m), ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"epoch {ep:2d} loss {np.mean(losses):.4f} val auroc {m.get('auroc', 0):.3f} "
              f"segF1@0.5 {m.get('segment_f1@0.5', 0):.3f} tIoU {m.get('mean_tiou', 0):.3f}", flush=True)
    model.load_state_dict(best[2])
    sv, vv, cv = infer(model, Xv, device)
    # thresholds tuned on val: clip decision (balanced accuracy) and frame decision (mean tIoU)
    yv = (Mv["class"] > 0).to_numpy()
    cands = np.quantile(sv, np.linspace(0.02, 0.98, 49))
    thr_clip = float(max(cands, key=lambda t: 0.5 * ((sv > t)[yv].mean() + (sv <= t)[~yv].mean())))
    thr_frame = float(max(np.linspace(0.2, 0.8, 13), key=lambda t: metrics.summarize(
        metrics.predictions(Mv, sv, vv, cv, fps_slot, thr_clip, t)).get("mean_tiou", 0)))
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    run = dict(features=features, pairwise=pairwise, epochs=epochs, lr=lr, seed=seed, best_epoch=best[1],
               d_in=int(X.shape[-1]), slots=int(X.shape[1]), fps_slot=int(fps_slot),
               thr_clip=thr_clip, thr_frame=thr_frame, history=history)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    print(f"best epoch {best[1]}, thresholds clip {thr_clip:.3f} frame {thr_frame:.2f} -> {out}")
    return run


def evaluate(root, run_dir, splits=None):
    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalHead(run["d_in"], run["slots"]).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    X, Y, M, fps_slot = data.load(root, run["features"], splits)
    results, md = {}, [f"# Results: {os.path.basename(run_dir)}", "",
                       f"features `{run['features']}`, pairwise loss {run['pairwise']}, best epoch {run['best_epoch']}, "
                       f"thresholds clip {run['thr_clip']:.3f} / frame {run['thr_frame']:.2f}", ""]
    tables = []
    for split in [s for s in ("val", "test_id", "test_ood_objects", "test_ood_env", "test_ood_scenario",
                              "test_ood_subtype", "test_renderer", "train") if s in set(M["split"])]:
        sel = (M["split"] == split).to_numpy()
        Ms = M[sel].reset_index(drop=True)
        s, v, c = infer(model, X[sel], device)
        P = metrics.predictions(Ms, s, v, c, fps_slot, run["thr_clip"], run["thr_frame"])
        results[split] = dict(summary=metrics.summarize(P),
                              breakdown={k: t.reset_index().to_dict("records") for k, t in metrics.breakdown(P).items()})
        P.drop(columns=["frame_labels"], errors="ignore").to_parquet(os.path.join(run_dir, f"predictions-{split}.parquet"))
        tables.append(pd.Series(results[split]["summary"], name=split))
        if split in ("test_id", "test_ood_subtype", "test_renderer"):
            for key, t in metrics.breakdown(P).items():
                md += [f"### {split}: by {key}", "", t.to_markdown(), ""]
    summary = pd.DataFrame(tables).T.round(3)
    md[4:4] = ["## Summary per split", "", summary.to_markdown(), ""]
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    with open(os.path.join(run_dir, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(summary.to_string())
    return results
