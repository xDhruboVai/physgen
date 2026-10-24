"""Automatic validators V1-V10 (doc 9.1). Each returns None when the check passes, or a reason."""
import numpy as np

from . import camera, shapes


def v1_stable(tr):
    if tr.unstable or not np.isfinite(tr.pos).all():
        return "V1 unstable simulation"


def v2_penetration(tr, tol=0.005):
    if tr.max_penetration > tol:
        return f"V2 penetration {tr.max_penetration * 1000:.1f} mm"


def key_ids(scene):
    return [o["id"] for o in scene["objects"] if o["role"] == "dynamic"]


def v3_visible(scene, tr, lib, frac=0.8, min_px=14.0):
    """Key objects in frame for most of the clip, and large enough to read (doc 6.2)."""
    T, S, fps = scene["clip"]["frames"], scene["clip"]["substeps"], scene["clip"]["fps"]
    W, H = scene["clip"]["resolution"]
    cam = scene["camera"]
    f = 1.0 / np.tan(np.radians(cam["fov_deg"]) / 2)
    sizes = []
    for i in key_ids(scene):
        r = float(np.linalg.norm(shapes.local_points(scene["objects"][i]["geom"], lib), axis=1).max())
        inside = 0
        for t in range(T):
            ndc, depth = camera.project(cam, t / fps, tr.pos[i, t * S][None], W / H)
            inside += bool(depth[0] > 0.05 and np.abs(ndc[0]).max() < 0.98)
            if t == 0:
                sizes.append(2 * r * f / depth[0] * H / 2)
        need = 0.25 if scene["objects"][i]["tag"] == "striker" else frac  # strikers may roll into view
        if inside < need * T:
            return f"V3 object {i} visible in {inside}/{T} frames"
    if np.median(sizes) < min_px or min(sizes) < 0.6 * min_px:
        return f"V3 objects too small ({min(sizes):.1f}px)"



def v4_motion(scene, tr, min_m):
    T, S = scene["clip"]["frames"], scene["clip"]["substeps"]
    ids = key_ids(scene)
    path = np.linalg.norm(np.diff(tr.pos[ids, :T * S + 1:S], axis=1), axis=2).sum()
    if path < min_m:
        return f"V4 too little motion ({path:.3f} m)"


def energy(scene, tr, inertia, steps):
    g = -scene["physics"]["gravity"]
    E = np.zeros(len(steps))
    for i in key_ids(scene):
        m = scene["objects"][i]["mass_kg"]
        R = shapes.quat_to_mats(tr.quat[i, steps])
        w_body = np.einsum("tji,tj->ti", R, tr.ang[i, steps])
        E += 0.5 * m * (tr.lin[i, steps] ** 2).sum(1) + 0.5 * (w_body ** 2 * inertia[i]).sum(1) + m * g * tr.pos[i, steps, 2]
    return E


def v7_energy(scene, tr, rel=0.1, abs_tol=0.003):
    """c0 total mechanical energy must not rise without an external cause."""
    T, S = scene["clip"]["frames"], scene["clip"]["substeps"]
    E = energy(scene, tr, tr.inertia, np.arange(0, T * S + 1, S))
    rise = (E - np.minimum.accumulate(E)).max()
    span = max(np.ptp(E), 1e-9)
    if rise > rel * span + abs_tol:
        return f"V7 energy rise {rise:.4f} J (span {span:.4f} J)"


def v5_identical(frames0, frames1, onset):
    """Violating clip must be pixel-identical to c0 before the onset (failure is a generator bug)."""
    if onset and not np.array_equal(frames0[:onset], frames1[:onset]):
        bad = int(np.nonzero((frames0[:onset] != frames1[:onset]).any((1, 2, 3)))[0][0])
        raise AssertionError(f"V5 frames differ before onset (first at {bad}, onset {onset})")


def v6_visible(frames0, frames1, onset, offset, threshold, level=10):
    d = np.abs(frames0[onset:offset].astype(np.int16) - frames1[onset:offset]).max(-1) > level
    frac = float(d.mean((1, 2)).max()) if len(d) else 0.0
    if frac < threshold:
        return f"V6 violation barely visible ({frac:.4f} < {threshold})", frac
    return None, frac


def v8_labels(meta):
    T = meta["num_frames"]
    lab, cls = np.asarray(meta["frame_labels"]), np.asarray(meta["frame_classes"])
    if len(lab) != T or len(cls) != T:
        raise AssertionError("V8 label length")
    if meta["class"] == 0:
        if lab.any() or cls.any() or meta["onset_frame"] is not None:
            raise AssertionError("V8 c0 has violation labels")
        return
    on, off = meta["onset_frame"], meta["offset_frame"]
    want = np.zeros(T, int)
    want[on:off] = 1
    if not (0 < on < off <= T) or not np.array_equal(lab, want) or not np.array_equal(cls, want * meta["class"]):
        raise AssertionError(f"V8 labels inconsistent with onset {on} offset {off}")


def scene_hash(scene):
    """V9: near-duplicate key (rounded layout and appearance)."""
    import hashlib
    key = [scene["scenario"], scene["environment"]["floor_texture"], scene["environment"]["hdri"]]
    for o in scene["objects"]:
        key += [o["shape"], o["texture"], *np.round(o["pos"], 2), *np.round(o["lin_vel"], 1)]
    return hashlib.sha1(repr(key).encode()).hexdigest()[:16]
