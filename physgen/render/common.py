"""Radiometry and timing shared by every renderer backend, so scene configs stay renderer-agnostic."""
import numpy as np


def kelvin_rgb(k):
    """Approximate black-body colour (linear sRGB, unit luminance)."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c / (c @ [0.2126, 0.7152, 0.0722])


def key_irradiance(scene):
    L = scene["light"]
    return L["intensity"] * kelvin_rgb(L["color_temp"])


def exposure(scene):
    """Analytic exposure: a mid floor reads as mid grey, then the baked EV jitter (D11)."""
    L, env = scene["light"], scene["environment"]
    e_floor = L["intensity"] * (np.sin(np.radians(L["elevation"])) + env["ambient"])
    albedo = 0.32
    return 0.25 / (albedo * e_floor / np.pi) * 2.0 ** env["exposure_ev"]


def sample_times(frame, cfg):
    """Sub-frame times (in frames) for baked motion blur."""
    K = cfg["renderer"]["motion_blur_samples"]
    return [frame + cfg["renderer"]["shutter"] * j / K for j in range(K)]


def frame_steps(frame, S, cfg):
    """Physics step indices that a frame's motion-blur samples interpolate between."""
    ts = sample_times(frame, cfg)
    return int(np.floor(ts[0] * S)), int(np.ceil(ts[-1] * S))


def interp_poses(pos, quat, s):
    """Poses at fractional step s: lerp positions, nlerp quaternions. pos (N,K,3), quat (N,K,4)."""
    k0 = int(np.floor(s))
    k1 = min(k0 + 1, pos.shape[1] - 1)
    a = s - k0
    if a == 0:
        return pos[:, k0], quat[:, k0]
    q0, q1 = quat[:, k0], quat[:, k1]
    q1 = np.where((np.sum(q0 * q1, -1) < 0)[:, None], -q1, q1)
    q = (1 - a) * q0 + a * q1
    return (1 - a) * pos[:, k0] + a * pos[:, k1], q / np.linalg.norm(q, axis=-1, keepdims=True)


def tonemap(lin):
    """ACES (Narkowicz) then sRGB encoding, float -> uint8."""
    x = np.maximum(lin, 0)
    y = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    s = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return np.round(np.clip(s, 0, 1) * 255).astype(np.uint8)


def model_matrices(pos, quat):
    from ..shapes import quat_to_mats
    M = np.zeros((len(pos), 4, 4))
    M[:, :3, :3] = quat_to_mats(quat)
    M[:, :3, 3] = pos
    M[:, 3, 3] = 1
    return M
