"""Stage timings for the performance table (doc 8.4). Run: python tests/benchmark.py"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from physgen import config, physics, pipeline, sampler, violations, writer  # noqa: E402
from physgen.assets import AssetLibrary  # noqa: E402
from physgen.render import get_renderer  # noqa: E402

cfg = config.load()
lib = AssetLibrary("assets")
T = cfg["dataset"]["clip"]["frames"]
scenes = [sampler.sample_scene("train", i, cfg, lib) for i in range(30)]
t0 = time.time()
for sc in scenes:
    physics.simulate(sc, T + pipeline.TAIL, mesh_lib=lib)
print(f"simulation: {len(scenes) / (time.time() - t0):.1f} c0 sims/s on one core")

ren = get_renderer(cfg, lib)
frames, secs, clips = 0, 0.0, []
for i in range(6):
    sc, tr = pipeline.simulate_c0("train", i, cfg, lib, [])
    prep = ren.prepare(sc, pipeline.shadow_bounds(sc, tr, lib, T))
    tk = violations.base_track(sc, tr)
    ren.render(prep, tr.pos, tr.quat, tk, [0])  # warm-up (texture upload)
    t0 = time.time()
    clips.append(ren.render(prep, tr.pos, tr.quat, tk, list(range(T))))
    secs += time.time() - t0
    frames += T
    ren.release(prep)
print(f"raster rendering: {frames / secs:.0f} frames/s ({cfg['renderer']['motion_blur_samples']} blur samples, "
      f"{cfg['dataset']['clip']['resolution']} px)")
t0 = time.time()
size = sum(sum(len(b) for b in writer.encode(c, cfg).values()) for c in clips)
print(f"encoding ({cfg['output']['format']}): {len(clips) / (time.time() - t0):.1f} clips/s, {size / len(clips) / 1e6:.2f} MB/clip")
