"""Evaluation protocol (doc section 10): clip-level detection, classification, temporal localisation."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

IOUS = (0.3, 0.5, 0.7)
BREAKDOWNS = ("class", "subtype", "severity", "scenario", "camera_mode", "timing")


def best_segment(p, thr):
    """Highest-mass connected run of frames with violation prob > thr: (onset, offset) or None.
    Each clip holds at most one violation, so one segment per clip."""
    on, best, best_mass = None, None, 0.0
    for t, v in enumerate(np.r_[p, 0.0]):
        if v > thr and on is None:
            on = t
        elif v <= thr and on is not None:
            mass = p[on:t].sum()
            if mass > best_mass:
                best, best_mass = (on, t), mass
            on = None
    return best


def predictions(meta, score, slot_viol, slot_cls, fps_slot, thr_clip, thr_frame):
    """Per-clip predictions: violation flag, class, segment (frames)."""
    rows = []
    for i in range(len(meta)):
        pf = np.repeat(slot_viol[i], fps_slot)
        seg = best_segment(pf, thr_frame)
        if seg:
            a, b = seg[0] // fps_slot, -(-seg[1] // fps_slot)
            cls = int(slot_cls[i, a:b].mean(0).argmax()) + 1
        else:
            cls = int(slot_cls[i].max(0).argmax()) + 1
        flag = bool(score[i] > thr_clip)
        rows.append(dict(score=float(score[i]), pred_violation=flag, pred_class=cls if flag else 0,
                         pred_onset=seg[0] if seg else None, pred_offset=seg[1] if seg else None))
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _iou(r):
    if r["class"] == 0 or r["pred_onset"] is None or pd.isna(r["pred_onset"]):
        return 0.0
    a, b = int(r["onset_frame"]), int(r["offset_frame"])
    c, d = int(r["pred_onset"]), int(r["pred_offset"])
    inter = max(0, min(b, d) - max(a, c))
    return inter / (max(b, d) - min(a, c))


def summarize(P):
    """All doc-10 metrics for a prediction table (one row per clip)."""
    y = (P["class"] > 0).astype(int)
    out = {"clips": len(P), "violating": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, P["score"]))
    out["accuracy"] = float((P["pred_violation"].astype(int) == y).mean())
    s = P.set_index("clip_id")["score"]
    v = P[P["class"] > 0]
    c0 = [pid[:pid.rfind("_c")] for pid in v["pair_id"]]  # "<base>_c0_c<k>" -> "<base>_c0"
    ranked = [sv > s[c] for sv, c in zip(v["score"], c0) if c in s.index]
    if ranked:
        out["pairwise_ranking_acc"] = float(np.mean(ranked))
    out["macro_f1_4class"] = float(f1_score(P["class"], P["pred_class"], labels=[0, 1, 2, 3], average="macro", zero_division=0))
    if len(v):
        iou = v.apply(_iou, axis=1)
        det = v[iou > 0]
        out["mean_tiou"] = float(iou.mean())
        out["localised_fraction"] = float((iou > 0).mean())
        if len(det):
            out["onset_mae_frames"] = float((det["pred_onset"].astype(int) - det["onset_frame"].astype(int)).abs().mean())
            out["offset_mae_frames"] = float((det["pred_offset"].astype(int) - det["offset_frame"].astype(int)).abs().mean())
        fp_c0 = int(((P["class"] == 0) & P["pred_onset"].notna() & P["pred_violation"]).sum())
        for t in IOUS:
            tp = int(((iou >= t) & (v["pred_class"] == v["class"])).sum())
            fp = fp_c0 + int((v["pred_onset"].notna() & v["pred_violation"]).sum()) - tp
            fn = len(v) - tp
            out[f"segment_f1@{t}"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    return out


def breakdown(P):
    """Per-group metrics on violating clips (with their c0 partners for ranking)."""
    s = P.set_index("clip_id")["score"]
    v = P[P["class"] > 0].copy()
    v["tiou"] = v.apply(_iou, axis=1)
    v["c0_score"] = [s.get(pid[:pid.rfind("_c")], np.nan) for pid in v["pair_id"]]
    v["ranked"] = v["score"] > v["c0_score"]
    v["class_ok"] = v["pred_class"] == v["class"]
    tables = {}
    for key in BREAKDOWNS:
        g = v.groupby(key)
        tables[key] = pd.DataFrame(dict(n=g.size(), ranking_acc=g["ranked"].mean(), detected=g["pred_violation"].mean(),
                                        class_acc=g["class_ok"].mean(), mean_tiou=g["tiou"].mean())).round(3)
    return tables
