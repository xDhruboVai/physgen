"""Command-line interface (doc 8.6).

python -m physgen fetch-assets
python -m physgen generate --config cfg.yaml --split train --num-scenes 5000 --workers 6 --out dataset_root
python -m physgen preview  --split train --index 0 [--scenario stack] --out previews
python -m physgen validate --root dataset_root
python -m physgen audit    --root dataset_root
python -m physgen export-features --root dataset_root --model MCG-NJU/videomae-base
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

from . import config, sampler

_W = {}


def _init_worker(cfg):
    from .assets import AssetLibrary
    from .render import get_renderer
    _W["cfg"] = cfg
    _W["lib"] = AssetLibrary(cfg["assets"]["root"])
    _W["renderer"] = get_renderer(cfg, _W["lib"])


def _run_scene(job):
    from . import pipeline, writer
    split, index, replay = job
    cfg = _W["cfg"]
    t0 = time.time()
    try:
        check = np.random.default_rng([sampler.scene_seed(split, index), 99])
        if replay:
            with open(replay) as f:
                replay = json.load(f)
        scene, clips, log = pipeline.generate_scene(split, index, cfg, _W["lib"], _W["renderer"], check, replay)
    except pipeline.Rejected as e:
        return dict(index=index, error=str(e), log=[])
    except AssertionError:
        raise
    except Exception:
        return dict(index=index, error=traceback.format_exc(), log=[])
    enc = [(meta, writer.encode(frames, cfg)) for meta, frames in clips]
    return dict(index=index, scene=scene, clips=enc, log=log, seconds=time.time() - t0)


def cmd_generate(a):
    import multiprocessing as mp
    from . import writer
    cfg = config.load(a.config, _overrides(a))
    root = a.out
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"config-{a.split}.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    from .assets import AssetLibrary
    lib = AssetLibrary(cfg["assets"]["root"])
    os.makedirs(os.path.join(root, "splits"), exist_ok=True)
    with open(os.path.join(root, "splits", "heldout.json"), "w") as f:  # doc 5.5: held-out asset lists
        json.dump(dict(asset_manifest=lib.manifest_hash, assets=sorted(i for i, m in lib.items.items() if m["heldout"]),
                       **cfg["splits"]), f, indent=1)
    n = a.num_scenes or cfg["dataset"]["num_scenes"]
    done = writer.committed_seeds(root, a.split)
    def replay(i):  # test_renderer re-renders stored test_id scenes with their violation choices (renderer transfer)
        p = os.path.join(root, "configs", "test_id", f"s{sampler.scene_seed('test_id', i):09d}.json")
        return p if a.split == "test_renderer" and os.path.exists(p) else None

    jobs = [(a.split, i, replay(i)) for i in range(a.start, a.start + n) if sampler.scene_seed(a.split, i) not in done]
    print(f"{len(jobs)} scenes to generate ({n - len(jobs)} already committed) -> {root}", flush=True)
    w = writer.ShardWriter(root, a.split, cfg["output"]["shard_max_clips"])
    workers = a.workers or cfg["runtime"]["workers"]
    stats = dict(scenes=0, clips=0, failed=0, t0=time.time())
    logf = open(os.path.join(root, "logs", f"generate-{a.split}-{w.run}.jsonl"), "w")
    try:
        if workers <= 1:
            _init_worker(cfg)
            results = map(_run_scene, jobs)
        else:
            pool = mp.Pool(workers, initializer=_init_worker, initargs=(cfg,))
            results = pool.imap_unordered(_run_scene, jobs)
        for r in results:
            for entry in r["log"]:
                logf.write(writer.to_json(entry) + "\n")
            if "error" in r:
                stats["failed"] += 1
                logf.write(writer.to_json(dict(index=r["index"], error=r["error"])) + "\n")
                continue
            w.write_scene(r["scene"], r["clips"])
            stats["scenes"] += 1
            stats["clips"] += len(r["clips"])
            if stats["scenes"] % 10 == 0 or stats["scenes"] == len(jobs):
                el = time.time() - stats["t0"]
                print(f"{stats['scenes']}/{len(jobs)} scenes, {stats['clips']} clips, {stats['failed']} failed, "
                      f"{3600 * (stats['clips'] - stats['scenes']) / el:.0f} pairs/h", flush=True)
    finally:
        w.close()
        logf.close()
        if workers > 1:
            pool.terminate()
    writer.rebuild_manifests(root)
    print(f"done: {stats['scenes']} scenes, {stats['clips']} clips, {stats['failed']} failed in "
          f"{time.time() - stats['t0']:.0f}s")


def cmd_preview(a):
    """Render one scene and write a contact sheet (PNG) and a side-by-side video (MP4) for inspection."""
    from PIL import Image, ImageDraw
    from . import pipeline
    from .assets import AssetLibrary
    from .render import get_renderer
    over = _overrides(a)
    if a.scenario:
        over = config.merge(over, {"scenarios": {k: float(k == a.scenario) for k in sampler.SCENARIOS}})
    cfg = config.load(a.config, over)
    lib = AssetLibrary(cfg["assets"]["root"])
    ren = get_renderer(cfg, lib)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    scene, clips, log = pipeline.generate_scene(a.split, a.index, cfg, lib, ren)
    print(f"scene {scene['scene_seed']} {scene['scenario']} in {time.time() - t0:.1f}s")
    for e in log:
        print("  rejected:", e)
    T = clips[0][1].shape[0]
    step = max(1, T // 16)
    th = 112
    rows = []
    for meta, frames in clips:
        tiles = []
        for t in range(0, T, step):
            im = Image.fromarray(frames[t]).resize((th, th))
            if meta["frame_labels"][t]:
                ImageDraw.Draw(im).rectangle([0, 0, th - 1, th - 1], outline=(255, 0, 0), width=3)
            tiles.append(np.asarray(im))
        rows.append(np.concatenate(tiles, 1))
        print(f"  {meta['clip_id']}: {meta['subtype']} {meta['severity']} {meta['timing']} "
              f"[{meta['onset_frame']}, {meta['offset_frame']}) visible={meta['visible_fraction']} {meta['violation_params']}")
    base = os.path.join(a.out, f"s{scene['scene_seed']:09d}_{scene['scenario']}")
    Image.fromarray(np.concatenate(rows, 0)).save(base + ".png")
    import imageio_ffmpeg
    side = []
    for t in range(T):
        row = []
        for meta, frames in clips:
            f = frames[t].copy()
            if meta["frame_labels"][t]:
                f[:3], f[-3:], f[:, :3], f[:, -3:] = (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)
            row.append(f)
        side.append(np.concatenate(row, 1))
    H, W = side[0].shape[:2]
    wr = imageio_ffmpeg.write_frames(base + ".mp4", (W, H), fps=scene["clip"]["fps"], macro_block_size=1,
                                     output_params=["-crf", "16"])
    wr.send(None)
    for f in side:
        wr.send(np.ascontiguousarray(f))
    wr.close()
    print("wrote", base + ".png", base + ".mp4")


def cmd_fetch_assets(a):
    from .fetch_assets import fetch, restore
    if os.path.exists(os.path.join(a.root, "manifest.json")) and not a.refresh:
        return restore(a.root)  # same assets as the committed manifest
    fetch(a.root, a.floors, a.objects, a.hdris)
    if a.ycb:
        from .meshes import fetch_ycb
        fetch_ycb(a.root, a.ycb)


def cmd_validate(a):
    from .audit import validate
    sys.exit(0 if validate(a.root) else 1)


def cmd_audit(a):
    from .audit import audit
    audit(a.root, probes=not a.no_probes)


def cmd_export_features(a):
    from .features import export
    export(a.root, a.model, a.window, a.stride, a.batch, a.device)


def _overrides(a):
    over = {}
    if getattr(a, "renderer", None):
        over = config.merge(over, {"renderer": {"backend": a.renderer}})
    if getattr(a, "assets", None):
        over = config.merge(over, {"assets": {"root": a.assets}})
    return over


def main(argv=None):
    p = argparse.ArgumentParser(prog="physgen")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config")
    g.add_argument("--split", default="train", choices=sampler.SPLITS)
    g.add_argument("--num-scenes", type=int)
    g.add_argument("--start", type=int, default=0, help="first scene index (runs with disjoint ranges can be merged)")
    g.add_argument("--workers", type=int)
    g.add_argument("--renderer", choices=["raster", "blender"])
    g.add_argument("--assets")
    g.add_argument("--out", default="dataset_root")
    g.set_defaults(fn=cmd_generate)
    v = sub.add_parser("preview")
    v.add_argument("--config")
    v.add_argument("--split", default="train", choices=sampler.SPLITS)
    v.add_argument("--index", type=int, default=0)
    v.add_argument("--scenario", choices=list(sampler.SCENARIOS))
    v.add_argument("--renderer", choices=["raster", "blender"])
    v.add_argument("--assets")
    v.add_argument("--out", default="previews")
    v.set_defaults(fn=cmd_preview)
    f = sub.add_parser("fetch-assets")
    f.add_argument("--root", default="assets")
    f.add_argument("--floors", type=int, default=48)
    f.add_argument("--objects", type=int, default=40)
    f.add_argument("--hdris", type=int, default=40)
    f.add_argument("--ycb", type=int, default=0, help="also fetch this many YCB meshes (0 = none)")
    f.add_argument("--refresh", action="store_true", help="re-pick assets from the live catalogue (changes the manifest)")
    f.set_defaults(fn=cmd_fetch_assets)
    va = sub.add_parser("validate")
    va.add_argument("--root", default="dataset_root")
    va.set_defaults(fn=cmd_validate)
    au = sub.add_parser("audit")
    au.add_argument("--root", default="dataset_root")
    au.add_argument("--no-probes", action="store_true")
    au.set_defaults(fn=cmd_audit)
    ef = sub.add_parser("export-features")
    ef.add_argument("--root", default="dataset_root")
    ef.add_argument("--model", default="MCG-NJU/videomae-base", help="any Hugging Face video backbone id")
    ef.add_argument("--window", type=int, default=16)
    ef.add_argument("--stride", type=int, default=8)
    ef.add_argument("--batch", type=int, default=8)
    ef.add_argument("--device", default="cuda")
    ef.set_defaults(fn=cmd_export_features)
    a = p.parse_args(argv)
    a.fn(a)
