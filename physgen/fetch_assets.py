"""Download CC0 textures and HDRIs from Poly Haven, make procedural patterns, write assets/manifest.json.

Usage: python -m physgen fetch-assets --root assets [--floors 48 --objects 40 --hdris 40]
"""
import hashlib
import json
import os
import urllib.request

import numpy as np
from PIL import Image

API = "https://api.polyhaven.com"
UA = {"User-Agent": "physgen-thesis-dataset/0.1"}


def _get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def _download(url, path, md5=None):
    if os.path.exists(path) and (md5 is None or _md5(path) == md5):
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        data = r.read()
    if md5 and hashlib.md5(data).hexdigest() != md5:
        raise IOError(f"md5 mismatch for {url}")
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _pick(assets, keep, n, seed):
    ids = sorted(i for i, a in assets.items() if keep(set(a.get("categories", []))))
    rng = np.random.default_rng(seed)
    return [ids[i] for i in rng.permutation(len(ids))[:n]]


def heldout(asset_id, fraction):
    """Stable per-asset holdout flag (design doc 5.6): independent of download order or count."""
    h = int(hashlib.sha1(asset_id.encode()).hexdigest()[:8], 16) / 2**32
    return h < fraction


def procedural_textures(out_dir, n=48, size=256, seed=1234):
    """Two-colour patterns so that spin of round objects is visible (doc 6.2)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    kinds = ["checker", "stripes", "dots", "noise", "grid", "rings", "zigzag", "blotch"]
    files = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        f = rng.integers(3, 9)
        if kind == "checker":
            m = ((np.floor(xx * f) + np.floor(yy * f)) % 2)
        elif kind == "stripes":
            m = (np.sin(2 * np.pi * f * (xx + yy * rng.uniform(-1, 1))) > 0)
        elif kind == "dots":
            m = (((xx * f) % 1 - .5) ** 2 + ((yy * f) % 1 - .5) ** 2) < rng.uniform(.04, .12)
        elif kind == "grid":
            m = (((xx * f) % 1) < .15) | (((yy * f) % 1) < .15)
        elif kind == "rings":
            m = np.sin(2 * np.pi * f * np.hypot(xx - .5, yy - .5)) > 0
        elif kind == "zigzag":
            m = ((yy * f + np.abs((xx * f) % 1 - .5)) % 1) < .5
        else:  # noise / blotch: tileable value noise from a few low-frequency sinusoids
            ph = rng.uniform(0, 2 * np.pi, (6, 3))
            v = sum(np.sin(2 * np.pi * (k + 1) * (xx * np.cos(p[0]) + yy * np.sin(p[0])) + p[1]) / (k + 1)
                    for k, p in enumerate(ph))
            m = v > (0 if kind == "blotch" else rng.uniform(-.3, .3))
        m = m.astype(np.float32)[..., None]
        c1, c2 = _rand_color(rng), _rand_color(rng)
        img = (m * c1 + (1 - m) * c2) * 255
        path = os.path.join(out_dir, f"proc_{i:03d}.png")
        Image.fromarray(img.astype(np.uint8)).save(path)
        files.append((f"proc_{i:03d}", path, kind))
    return files


def _rand_color(rng):
    import colorsys
    h, s, v = rng.uniform(), rng.uniform(.3, .9), rng.uniform(.25, .95)
    return np.array(colorsys.hsv_to_rgb(h, s, v), np.float32)


def restore(root="assets"):
    """Re-download exactly the assets listed in a committed manifest.json (reproducible datasets:
    a fresh `fetch` re-picks from Poly Haven's live catalogue, which changes over time)."""
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    missing = [m for m in manifest if not os.path.exists(os.path.join(root, m["file"]))]
    if any(m["source"] == "procedural" for m in missing):
        procedural_textures(os.path.join(root, "procedural"))
    for m in missing:
        if m["source"] == "polyhaven":
            os.makedirs(os.path.dirname(os.path.join(root, m["file"])), exist_ok=True)
            print("fetch", m["file"])
            _download(m["url"], os.path.join(root, m["file"]), m.get("md5"))
        elif m["source"] == "ycb" and m["kind"] == "mesh":
            from .meshes import ycb_one
            ycb_one(root, m["id"][len("ycb_"):])
    still = [m["file"] for m in manifest if not os.path.exists(os.path.join(root, m["file"]))]
    if still:
        raise IOError(f"{len(still)} assets could not be restored, e.g. {still[:3]}")
    print(f"{len(manifest)} assets present ({len(missing)} restored)")


def fetch(root="assets", n_floor=48, n_object=40, n_hdri=40, held_out_fraction=0.15):
    os.makedirs(root, exist_ok=True)
    manifest = []
    tex = _get_json(f"{API}/assets?t=textures")
    hdr = _get_json(f"{API}/assets?t=hdris")
    bad = {"aerial", "terrain", "snow", "rock", "sand", "gravel", "road", "asphalt", "bark", "collection: moon"}
    floors = _pick(tex, lambda c: "floor" in c and not c & bad, n_floor, 0)
    objects = _pick(tex, lambda c: c & {"fabric", "wood", "metal", "man made", "plastic"}
                    and not c & (bad | {"floor", "wall", "roofing", "brick"}), n_object, 1)
    hdris = _pick(hdr, lambda c: c & {"indoor", "studio", "overcast"} and not c & {"high contrast", "night", "skies"},
                  n_hdri, 2)
    jobs = [("floor", i) for i in floors] + [("object", i) for i in objects] + [("hdri", i) for i in hdris]
    for kind, aid in jobs:
        files = _get_json(f"{API}/files/{aid}")
        if kind == "hdri":
            entry, sub, ext = files["hdri"]["1k"]["hdr"], "hdris", "hdr"
        else:
            key = next((k for k in files if k.lower() in ("diffuse", "diff", "color", "albedo")), None)
            if key is None or "jpg" not in files[key].get("1k", {}):
                print("skip", aid, "(no 1k diffuse jpg)")
                continue
            entry, sub, ext = files[key]["1k"]["jpg"], "textures", "jpg"
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        rel = f"{sub}/{aid}.{ext}"
        print("fetch", rel, entry["size"])
        _download(entry["url"], os.path.join(root, rel), entry.get("md5"))
        manifest.append(dict(id=f"ph_{aid}", kind=kind, file=rel, source="polyhaven", license="CC0",
                             url=entry["url"], md5=entry.get("md5")))
    for pid, path, pkind in procedural_textures(os.path.join(root, "procedural")):
        manifest.append(dict(id=pid, kind="object", file=os.path.relpath(path, root).replace("\\", "/"),
                             source="procedural", license="CC0", pattern=pkind))
    for m in manifest:
        m["heldout"] = heldout(m["id"], held_out_fraction)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(manifest)} assets, {sum(m['heldout'] for m in manifest)} held out")
    return manifest
