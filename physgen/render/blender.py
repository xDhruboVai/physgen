"""Blender backend (D1 options C/D): re-renders the same scene configs, poses and light track with
Cycles or EEVEE in a Blender subprocess (blender_worker.py), then applies the shared exposure and
tone mapping. Slower than the raster backend; intended for the test_renderer evaluation subset."""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from .. import camera, shapes
from . import common

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_worker.py")


class BlenderRenderer:
    name = "blender"

    def __init__(self, cfg, lib):
        self.cfg, self.lib = cfg, lib
        b = cfg["renderer"]["blender"]
        self.exe = b["executable"] if os.path.isabs(b["executable"]) else os.path.join(ROOT, b["executable"])
        if not os.path.exists(self.exe):
            raise FileNotFoundError(f"Blender not found at {self.exe} (renderer.blender.executable)")
        self.tmp = os.path.join(ROOT, "third_party", "tmp", "blender")
        os.makedirs(self.tmp, exist_ok=True)

    def prepare(self, scene, bounds):
        lib, env, L = self.lib, scene["environment"], scene["light"]
        d = tempfile.mkdtemp(dir=self.tmp)
        objects = []
        for o in scene["objects"]:
            p, n, uv = shapes.mesh_arrays(o["geom"], lib)
            path = os.path.join(d, f"geo{o['id']}.npz")
            np.savez(path, positions=p, normals=n, uvs=uv)
            objects.append(dict(geometry=path, texture=lib.path(o["texture"]), mode=o.get("tex_mode", "triplanar"),
                                tex_scale=o["tex_scale_m"], tint=o["tint"], mean=lib.texture(o["texture"])[1].tolist(),
                                contrast=1.0, gain=1.0, roughness=o["roughness"], metallic=o["metallic"]))
        kc = common.kelvin_rgb(L["color_temp"])
        _, _, hdri_scale = lib.hdri(env["hdri"])
        b, r = self.cfg["renderer"]["blender"], self.cfg["renderer"]
        job = dict(resolution=scene["clip"]["resolution"], fps=scene["clip"]["fps"], substeps=scene["clip"]["substeps"],
                   shutter=r["shutter"], blur_samples=r["motion_blur_samples"],
                   render=dict(engine=b["engine"], device=b["device"], samples=b["samples"], denoise=b["denoise"]),
                   hdri=lib.path(env["hdri"]), hdri_rot=float(np.radians(env["hdri_rot_deg"])),
                   world_strength=float(hdri_scale * env["ambient"] * L["intensity"] * kc.max()),
                   light_color=(kc / kc.max()).tolist(), light_energy=float(L["intensity"] * kc.max()),
                   floor=dict(centre=[float(x) for x in bounds[0][:2]], texture=lib.path(env["floor_texture"]),
                              mode="planar", tex_scale=env["floor_uv_m"], tint=[1.0, 1.0, 1.0],
                              mean=lib.texture(env["floor_texture"])[1].tolist(), contrast=env["floor_contrast"],
                              gain=env["floor_gain"], roughness=env["floor_roughness"], metallic=0.0),
                   fov_deg=scene["camera"]["fov_deg"], objects=objects)
        with open(os.path.join(d, "job.json"), "w") as f:
            json.dump(job, f)
        return dict(dir=d, scene=scene, n=0)

    def release(self, prep):
        shutil.rmtree(prep["dir"], ignore_errors=True)

    def render(self, prep, pos, quat, track, frames):
        scene = prep["scene"]
        S, fps = scene["clip"]["substeps"], scene["clip"]["fps"]
        k0 = frames[0] * S
        k1 = min((frames[-1] + 1) * S + 1, pos.shape[1])
        steps = np.arange(k0, k1)
        eyes, quats = [], []
        for k in steps:
            eye, target = camera.pose(scene["camera"], k / S / fps)
            R = camera.view_matrix(eye, target)[:3, :3].T  # camera-to-world: Blender cameras look along -Z
            eyes.append(eye)
            quats.append(Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]])
        angle = 2 * np.asarray(track["softness"], float)
        if "mode_obj" in track:  # S1.6 has no per-pixel equivalent here: drive the light size from the object height
            for f in np.nonzero(track["mode"])[0]:
                h = max(float(pos[track["mode_obj"], f * S, 2]), 0.02)
                t = np.tan(track["softness"][f]) * (0.08 if track["mode"][f] == 1 else max(0.4 - h, 0.0)) / h
                angle[f] = 2 * np.arctan(t)
        prep["n"] += 1
        clip = os.path.join(prep["dir"], f"clip{prep['n']}.npz")
        out = os.path.join(prep["dir"], f"out{prep['n']}.npy")
        np.savez(clip, k0=k0, frames=np.asarray(frames), pos=pos[:, k0:k1], quat=quat[:, k0:k1],
                 spos=track["spos"][:, k0:k1], squat=track["squat"][:, k0:k1], shade_dir=track["shade_dir"],
                 shadow_dir=track["shadow_dir"], sun_angle=angle, cam_eye=np.array(eyes), cam_quat=np.array(quats))
        res = subprocess.run([self.exe, "-b", "--factory-startup", "-noaudio", "-P", WORKER, "--",
                              os.path.join(prep["dir"], "job.json"), clip, out], capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"Blender failed ({res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
        lin = np.load(out)
        return common.tonemap(lin * common.exposure(scene))
