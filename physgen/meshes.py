"""Real meshes (D7 option B): YCB objects, preprocessed once and cached (doc 6.2).

Each mesh is centred at its centre of mass and rotated to its principal axes (so PyBullet's
inertia is diagonal), convex-decomposed with VHACD for collisions, and keeps its UV texture.
"""
import json
import os
import tarfile
import urllib.request

import numpy as np

from .fetch_assets import UA, heldout

YCB_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/{}_google_16k.tgz"
# rigid, not thin, not tiny (doc 6.2 exclusions)
YCB = ["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle",
       "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana",
       "013_apple", "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum", "019_pitcher_base",
       "021_bleach_cleanser", "024_bowl", "025_mug", "026_sponge", "035_power_drill", "036_wood_block",
       "040_large_marker", "048_hammer", "053_mini_soccer_ball", "054_softball", "055_baseball", "056_tennis_ball",
       "057_racquetball", "061_foam_brick", "062_dice", "065-a_cups", "065-d_cups", "077_rubiks_cube"]


def read_obj(path):
    """Triangle soup (positions, uvs, normals) from a triangulated OBJ (v/vt/vn faces).
    shortcut: no polygons > 3 or negative indices; YCB google_16k files are plain triangles."""
    v, vt, vn, f = [], [], [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                v.append([float(x) for x in t[1:4]])
            elif t[0] == "vt":
                vt.append([float(x) for x in t[1:3]])
            elif t[0] == "vn":
                vn.append([float(x) for x in t[1:4]])
            elif t[0] == "f":
                f.append([[int(i) - 1 if i else -1 for i in (c.split("/") + ["", ""])[:3]] for c in t[1:4]])
    v, f = np.array(v), np.array(f)
    pos = v[f[..., 0]]
    uv = np.array(vt)[f[..., 1]] if vt and (f[..., 1] >= 0).all() else np.zeros(pos.shape[:2] + (2,))
    if vn and (f[..., 2] >= 0).all():
        nrm = np.array(vn)[f[..., 2]]
    else:
        n = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
        nrm = np.repeat((n / np.linalg.norm(n, axis=1, keepdims=True))[:, None], 3, 1)
    return pos, uv, nrm


def preprocess(obj_path, out_dir, name):
    import pybullet as p
    from . import shapes
    tris, uv, nrm = read_obj(obj_path)
    vol, com, inertia = shapes.mass_props(tris)
    if vol <= 0 or not np.isfinite(com).all():  # not closed: fall back to the convex hull's mass properties
        vol, com, inertia = shapes.mass_props(shapes.hull_tris(tris.reshape(-1, 3)))
    w, R = np.linalg.eigh(inertia)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    tris = (tris - com) @ R  # body frame = principal axes at the centre of mass
    nrm = nrm @ R
    np.savez(os.path.join(out_dir, f"{name}.npz"), positions=tris.reshape(-1, 3).astype(np.float32),
             normals=nrm.reshape(-1, 3).astype(np.float32), uvs=uv.reshape(-1, 2).astype(np.float32))
    body_obj = os.path.join(out_dir, f"{name}_body.obj")
    with open(body_obj, "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in tris.reshape(-1, 3))
        fh.writelines(f"f {3 * i + 1} {3 * i + 2} {3 * i + 3}\n" for i in range(len(tris)))
    coll = os.path.join(out_dir, f"{name}_vhacd.obj")
    if not os.path.exists(coll):
        cid = p.connect(p.DIRECT)
        p.vhacd(body_obj, coll, os.path.join(out_dir, f"{name}_vhacd.log"), resolution=200000, physicsClientId=cid)
        p.disconnect(cid)
    extent = float(np.ptp(tris.reshape(-1, 3), axis=0).max())
    return dict(volume=float(vol), extent=extent, upright=shapes.mat_to_quat(R).tolist())


def ycb_one(root, name):
    """Download, unpack and preprocess one YCB object; returns its mesh info."""
    out = os.path.join(root, "ycb")
    os.makedirs(out, exist_ok=True)
    tgz = os.path.join(out, f"{name}.tgz")
    if not os.path.exists(tgz):
        print("fetch", name)
        with urllib.request.urlopen(urllib.request.Request(YCB_URL.format(name), headers=UA), timeout=300) as r:
            data = r.read()
        with open(tgz + ".part", "wb") as f:
            f.write(data)
        os.replace(tgz + ".part", tgz)
    src = os.path.join(out, name, "google_16k")
    if not os.path.exists(src):
        with tarfile.open(tgz) as t:
            t.extractall(out, filter="data")
    return preprocess(os.path.join(src, "textured.obj"), out, name)


def fetch_ycb(root, n=len(YCB), held_out_fraction=0.15):
    mpath = os.path.join(root, "manifest.json")
    with open(mpath) as f:
        manifest = [m for m in json.load(f) if m.get("source") != "ycb"]
    for name in YCB[:n]:
        info = ycb_one(root, name)
        if not os.path.exists(tgz):
            print("fetch", name)
            with urllib.request.urlopen(urllib.request.Request(YCB_URL.format(name), headers=UA), timeout=300) as r:
                data = r.read()
            with open(tgz + ".part", "wb") as f:
                f.write(data)
        tex_id = f"ycbtex_{name}"
        manifest.append(dict(id=tex_id, kind="meshtex", file=f"ycb/{name}/google_16k/texture_map.png", source="ycb",
                             license="CC-BY-4.0", url=YCB_URL.format(name), heldout=False))
        manifest.append(dict(id=f"ycb_{name}", kind="mesh", file=f"ycb/{name}.npz", collision=f"ycb/{name}_vhacd.obj",
                             texture=tex_id, source="ycb", license="CC-BY-4.0", url=YCB_URL.format(name),
                             heldout=heldout(f"ycb_{name}", held_out_fraction), **info))
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{n} YCB meshes processed")
