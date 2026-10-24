"""Asset library (doc 8.2 'assets'): textures, HDRIs, meshes, holdout flags, cached decoding."""
import functools
import hashlib
import json
import os

import numpy as np


class AssetLibrary:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        with open(os.path.join(self.root, "manifest.json")) as f:
            self.items = {m["id"]: m for m in json.load(f)}
        with open(os.path.join(self.root, "manifest.json"), "rb") as f:
            self.manifest_hash = hashlib.sha1(f.read()).hexdigest()[:12]

    def ids(self, kind, heldout=None, source=None):
        """heldout: None = any, False = training assets only, True = held-out only."""
        return sorted(i for i, m in self.items.items() if m["kind"] == kind
                      and (heldout is None or m["heldout"] == heldout) and (source is None or m["source"] == source))

    def path(self, aid):
        return os.path.join(self.root, self.items[aid]["file"])

    @functools.lru_cache(maxsize=32)
    def texture(self, aid):
        """sRGB uint8 RGB image and its mean colour in linear RGB."""
        from PIL import Image
        img = np.asarray(Image.open(self.path(aid)).convert("RGB"))
        f = img.astype(np.float32) / 255.0
        lin = np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)
        return img, lin.reshape(-1, 3).mean(0)

    def texture_mean(self, aid):
        """Mean linear luminance (used to normalise floor albedo, doc 6.1)."""
        return float(self.texture(aid)[1] @ [0.2126, 0.7152, 0.0722])

    @functools.lru_cache(maxsize=64)
    def hdri(self, aid):
        """Equirect radiance (H, W, 3) float32 and 9 irradiance SH coefficients, normalised so the
        irradiance on an upward-facing surface is 1 (doc 6.1: HDRI ambient)."""
        import cv2
        img = cv2.imread(self.path(aid), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
        img = np.nan_to_num(np.clip(img, 0, 1e4))
        sh = irradiance_sh(cv2.resize(img, (128, 64), interpolation=cv2.INTER_AREA))
        up = sh_eval(sh, np.array([0.0, 0.0, 1.0]))
        scale = 1.0 / max(float(up @ [0.2126, 0.7152, 0.0722]), 1e-6)
        return img * scale, sh * scale, scale

    @functools.lru_cache(maxsize=128)
    def load(self, mesh_id):
        """Mesh for rendering (centred at the centre of mass) plus its convex-decomposition file."""
        m = self.items[mesh_id]
        data = np.load(os.path.join(self.root, m["file"]))
        return dict(positions=data["positions"], normals=data["normals"], uvs=data["uvs"],
                    collision_file=os.path.join(self.root, m["collision"]), upright=m["upright"],
                    texture=m.get("texture"), extent=float(m["extent"]), volume=float(m["volume"]))


# Equirect convention shared by both renderers: direction d (z up), u = atan2(d.y, d.x)/(2pi)+0.5, v = acos(d.z)/pi.
def equirect_dirs(h, w):
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    th = v[:, None] * np.pi
    ph = (u[None, :] - 0.5) * 2 * np.pi
    d = np.stack(np.broadcast_arrays(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)), -1)
    solid = (np.sin(th) * (np.pi / h) * (2 * np.pi / w)) * np.ones_like(ph)
    return d, solid


def _sh_basis(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                     0.546274 * (x * x - y * y)], -1)


def irradiance_sh(img):
    """Ramamoorthi-Hanrahan irradiance SH (9 x RGB) of an equirect radiance map."""
    d, solid = equirect_dirs(*img.shape[:2])
    Y = _sh_basis(d)
    L = np.einsum("hwk,hwc,hw->kc", Y, img, solid)
    band = np.array([np.pi] + [2 * np.pi / 3] * 3 + [np.pi / 4] * 5)[:, None]
    return (L * band).astype(np.float32)


def sh_eval(sh, n):
    return _sh_basis(np.asarray(n, float)) @ sh
