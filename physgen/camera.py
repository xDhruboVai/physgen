"""Camera model shared by both renderers, the framing step and the visibility validator (V3)."""
import numpy as np


def pose(cam, t):
    """Eye and target at time t (seconds). Modes: static | orbit | pan (doc 6.5, D9)."""
    yaw = np.radians(cam["yaw_deg"] + cam.get("orbit_deg_per_s", 0.0) * t)
    pitch = np.radians(cam["pitch_deg"])
    target = np.asarray(cam["target"], float) + np.asarray(cam.get("pan_m_per_s", [0, 0, 0]), float) * t
    d = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    return target + cam["distance"] * d, target


def view_matrix(eye, target, up=(0, 0, 1)):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_deg, aspect, near=0.02, far=60.0):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def project(cam, t, pts, aspect=1.0):
    """World points (N,3) -> NDC xy (N,2) and view depth (N,)."""
    eye, target = pose(cam, t)
    V = view_matrix(eye, target)
    P = perspective(cam["fov_deg"], aspect)
    h = np.c_[pts, np.ones(len(pts))] @ (P @ V).T
    depth = h[:, 3]
    return h[:, :2] / np.maximum(depth[:, None], 1e-6), depth


def frame_points(pts, rng, cfg, aspect=1.0):
    """Choose a camera that frames all action points (doc 6.5: action always in view)."""
    c = cfg["camera"]
    lo, hi = pts.min(0), pts.max(0)
    target = (lo + hi) / 2
    moving = rng.uniform() < c["moving_fraction"]
    yaw = rng.uniform(0, 360)
    xy = pts[:, :2] - pts[:, :2].mean(0)
    w, V = np.linalg.eigh(xy.T @ xy)
    if w[1] > 1.7 * w[0]:  # elongated action (chains, rolls): look from the side, +-60 deg, so it is not foreshortened
        yaw = np.degrees(np.arctan2(V[1, 1], V[0, 1])) + 90 + rng.uniform(-60, 60) + 180 * rng.integers(2)
    cam = dict(mode="static", target=target.tolist(), yaw_deg=float(yaw % 360),
               pitch_deg=float(rng.uniform(*c["pitch_deg"])), fov_deg=float(rng.uniform(*c["fov_deg"])))
    fill = float(rng.uniform(*c["fill"]))
    if moving:
        if rng.uniform() < 0.5:
            cam.update(mode="orbit", orbit_deg_per_s=float(rng.choice([-1, 1]) * rng.uniform(3, 10)))
        else:
            a = rng.uniform(0, 2 * np.pi)
            cam.update(mode="pan", pan_m_per_s=[float(np.cos(a) * 0.04), float(np.sin(a) * 0.04), 0.0])
        fill *= 0.85
    R = np.linalg.norm(pts - target, axis=1).max() + 1e-3
    cam["distance"] = float(R / np.sin(np.radians(cam["fov_deg"]) / 2))
    for _ in range(6):  # tighten: scale distance so the projected extent matches `fill`, recentre
        ndc, depth = project(cam, 0.0, pts, aspect)
        ext = np.abs(ndc).max()
        centre = (ndc.max(0) + ndc.min(0)) / 2
        eye, tgt = pose(cam, 0.0)
        V = view_matrix(eye, tgt)
        f = 1.0 / np.tan(np.radians(cam["fov_deg"]) / 2)
        shift = (V[0, :3] * centre[0] * aspect + V[1, :3] * centre[1]) * cam["distance"] / f
        cam["target"] = (np.asarray(cam["target"]) + shift).tolist()
        cam["distance"] = float(max(cam["distance"] * (ext / fill) ** 0.7, 0.25))
    eye, _ = pose(cam, 0.0)
    if eye[2] < 0.08:
        cam["pitch_deg"] = float(np.degrees(np.arcsin(min(1, 0.1 / cam["distance"]))) + 5)
    return cam
