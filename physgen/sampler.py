"""Scene sampler (doc section 7): seed -> scene config. Hierarchical order:
scenario -> environment -> objects -> physical parameters -> light (camera and distractors are
chosen after the c0 simulation, from the same seed, in pipeline.py).

Every drawn value is stored in the config, so a clip is a pure function of
(scene seed, generator version, asset manifest).
"""
import numpy as np
from scipy.stats import qmc

from . import __commit__, __version__, shapes

SPLITS = ["train", "val", "test_id", "test_ood_objects", "test_ood_env", "test_ood_scenario",
          "test_ood_subtype", "test_renderer"]
# test_renderer re-renders the test_id scene configs with another renderer (doc 5.6, D1 = D).
SEED_BASE = {"train": 0, "val": 1, "test_id": 2, "test_renderer": 2, "test_ood_objects": 3,
             "test_ood_env": 4, "test_ood_scenario": 5, "test_ood_subtype": 6}
STREAMS = dict(scenario=1, env=2, objects=3, light=4, camera=5, violation=6, clutter=7)

MATERIALS = {  # doc 6.2 (starting values)
    "foam": dict(density=(50, 150), friction=(0.6, 0.9), restitution=(0.05, 0.25)),
    "wood": dict(density=(500, 800), friction=(0.3, 0.6), restitution=(0.2, 0.5)),
    "plastic": dict(density=(900, 1200), friction=(0.2, 0.5), restitution=(0.3, 0.7)),
    "rubber": dict(density=(1000, 1200), friction=(0.7, 1.0), restitution=(0.6, 0.9)),
    "metal": dict(density=(2700, 7800), friction=(0.15, 0.4), restitution=(0.1, 0.4)),
}

# scenario -> (compatible classes, c3 subtype, shapes); doc 6.4
SCENARIOS = {
    "free_fall": ({1, 2}, None, ["sphere", "box", "cylinder", "capsule", "wedge", "brick", "hull", "mesh"]),
    "ramp_slide": ({1, 2, 3}, "S3.4", ["box", "brick", "wedge", "cylinder", "capsule"]),
    "rolling": ({1, 2}, None, ["sphere", "cylinder", "capsule"]),
    "billiards": ({1, 2, 3}, "S3.1", ["sphere", "cylinder", "capsule"]),
    "stack": ({1, 2, 3}, "S3.3", ["box", "brick"]),
    "dominoes": ({1, 2, 3}, "S3.2", ["domino"]),
    "projectile_bin": ({1, 2}, None, ["sphere", "box", "cylinder", "capsule", "wedge", "brick", "hull", "mesh"]),
}


def scene_seed(split, index):
    return SEED_BASE[split] * 10 ** 8 + index


def rng_for(seed, stream, attempt=0):
    """Independent random stream per module (doc 8.3); `attempt` re-draws a rejected scene."""
    return np.random.default_rng(np.random.SeedSequence([seed, STREAMS[stream], attempt]))


def sobol_point(split, index, d=8):
    """Low-discrepancy draw for the top-level continuous parameters (doc 7)."""
    eng = qmc.Sobol(d, scramble=True, seed=SEED_BASE[split])
    if index:
        eng.fast_forward(index)
    return eng.random(1)[0]


def policy(split, cfg):
    """What a split may use (doc 5.6)."""
    s = cfg["splits"]
    p = dict(assets_heldout=False, shapes=None, scenarios=None, subtypes_heldout=False,
             heldout_shapes=set(s["heldout_shapes"]), heldout_scenarios=set(s["heldout_scenarios"]),
             heldout_subtypes=set(s["heldout_subtypes"]), meshes_heldout=False)
    if split == "test_ood_objects":
        p.update(shapes=p["heldout_shapes"] | {"mesh"}, meshes_heldout=True)
    elif split == "test_ood_env":
        p.update(assets_heldout=True)
    elif split == "test_ood_scenario":
        p.update(scenarios=p["heldout_scenarios"])
    elif split == "test_ood_subtype":
        p.update(subtypes_heldout=True)
    return p


def allowed_shapes(scenario, pol, cfg, lib):
    shapes_ = [s for s in SCENARIOS[scenario][2] if s != "mesh" or ("ycb" in cfg["assets"]["meshes"] and lib.ids("mesh"))]
    if pol["shapes"] is not None:
        return [s for s in shapes_ if s in pol["shapes"]]
    return [s for s in shapes_ if s not in pol["heldout_shapes"]]


def choose_scenario(rng, pol, cfg, lib):
    names, w = [], []
    for name, weight in cfg["scenarios"].items():
        if weight <= 0:
            continue
        if pol["scenarios"] is not None and name not in pol["scenarios"]:
            continue
        if pol["scenarios"] is None and name in pol["heldout_scenarios"]:
            continue
        if not allowed_shapes(name, pol, cfg, lib):
            continue
        names.append(name)
        w.append(weight)
    if not names:
        raise ValueError("no scenario is allowed by this config/split")
    w = np.array(w) / sum(w)
    return names[rng.choice(len(names), p=w)]


# ---------------------------------------------------------------- helpers
def yaw_quat(a):
    return [0.0, 0.0, float(np.sin(a / 2)), float(np.cos(a / 2))]


def axis_quat(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return [*(axis * np.sin(a / 2)).tolist(), float(np.cos(a / 2))]


def random_quat(rng):
    q = rng.normal(size=4)
    return (q / np.linalg.norm(q)).tolist()


def rot(q):
    return shapes.quat_to_mat(q)


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


class Builder:
    def __init__(self, rng, lib, pol, cfg):
        self.rng, self.lib, self.pol, self.cfg = rng, lib, pol, cfg
        self.objects = []

    def material(self, name=None, min_density=0.0):
        rng = self.rng
        min_density = min(min_density, 0.9 * MATERIALS["metal"]["density"][1])
        if name is None:
            names = [n for n, m in MATERIALS.items() if m["density"][1] >= min_density]
            name = names[rng.integers(len(names))]
        m = MATERIALS[name]
        lo = max(m["density"][0], min_density)
        return name, dict(density=float(rng.uniform(lo, m["density"][1])), friction=float(rng.uniform(*m["friction"])),
                          restitution=float(rng.uniform(*m["restitution"])))

    def texture(self, round_shape):
        """Appearance is independent of material (doc 6.2); round objects always get a pattern."""
        rng, lib = self.rng, self.lib
        held = self.pol["assets_heldout"]
        proc = lib.ids("object", heldout=held, source="procedural")
        real = lib.ids("object", heldout=held, source="polyhaven")
        pool = proc if (round_shape or not real or rng.uniform() < 0.5) else real
        return pool[rng.integers(len(pool))]

    def geom(self, shape, size):
        rng = self.rng
        if shape == "sphere":
            return dict(shape=shape, radius=size / 2), [0, 0, 0, 1]
        if shape == "box":
            dims = size * np.array([1.0, rng.uniform(0.45, 1), rng.uniform(0.35, 1)])
            return dict(shape=shape, half=(rng.permutation(dims) / 2).tolist()), [0, 0, 0, 1]
        if shape == "brick":
            dims = size * np.array([1.0, 0.5, 0.32]) * rng.uniform(0.9, 1.1, 3)
            return dict(shape=shape, half=(dims / 2).tolist()), [0, 0, 0, 1]
        if shape == "domino":
            return dict(shape=shape, half=[size * 0.08, size * 0.25, size / 2]), [0, 0, 0, 1]
        if shape == "cylinder":
            r = size / 2 * rng.uniform(0.35, 1.0)
            h = size if 2 * r < size * 0.99 else size * rng.uniform(0.3, 1.0)
            return dict(shape=shape, radius=float(r), height=float(h)), [0, 0, 0, 1]
        if shape == "capsule":
            r = size * rng.uniform(0.15, 0.3)
            return dict(shape=shape, radius=float(r), height=float(size - 2 * r)), [0, 0, 0, 1]
        if shape == "wedge":
            half = size / 2 * np.array([1.0, rng.uniform(0.4, 1.0), rng.uniform(0.3, 0.8)])
            pts, up = shapes.canonical_convex(shapes.wedge_points(half))
            return dict(shape=shape, points=pts.round(6).tolist()), up.tolist()
        if shape == "hull":
            ax = size / 2 * np.array([1.0, rng.uniform(0.5, 1.0), rng.uniform(0.4, 1.0)])
            d = rng.normal(size=(rng.integers(8, 15), 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            pts, up = shapes.canonical_convex(d * ax * rng.uniform(0.85, 1.0, (len(d), 1)))
            return dict(shape=shape, points=pts.round(6).tolist()), up.tolist()
        if shape == "mesh":
            ids = self.lib.ids("mesh", heldout=self.pol["meshes_heldout"])
            mid = ids[rng.integers(len(ids))]
            m = self.lib.load(mid)
            return dict(shape=shape, mesh=mid, scale=float(size / m["extent"])), m["upright"]
        raise ValueError(shape)

    def obj(self, shape, size, role="dynamic", tag="main", material=None, min_density=0.0, texture=None):
        g, upright = self.geom(shape, size)
        mname, mp = self.material(material, min_density)
        vol = self.lib.load(g["mesh"])["volume"] * g["scale"] ** 3 if shape == "mesh" else shapes.volume(g)
        rng = self.rng
        o = dict(id=len(self.objects), role=role, tag=tag, shape=shape, geom=g, upright=upright,
                 scale_cm=round(size * 100, 2), material=mname, density=mp["density"],
                 mass_kg=float(mp["density"] * vol), friction=mp["friction"], restitution=mp["restitution"],
                 rolling_friction=float(rng.uniform(0.0005, 0.003) * size) if shape in shapes.ROUND else 0.0,
                 spinning_friction=float(rng.uniform(0.0005, 0.003) * size) if shape in shapes.ROUND else 0.0,
                 texture=texture or self.texture(shape in shapes.ROUND), tint=self.tint(),
                 tex_scale_m=float(size * rng.uniform(0.5, 1.5)),
                 roughness=float(rng.uniform(0.25, 0.9)), metallic=float(rng.uniform() < 0.15),
                 pos=[0, 0, 0], quat=[0, 0, 0, 1], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0])
        if shape == "mesh":  # real meshes keep their own UV texture; material is still sampled independently
            o.update(texture=self.lib.load(g["mesh"])["texture"], tex_mode="uv", tint=[1.0, 1.0, 1.0], metallic=0.0)
        if size < 0.05:
            o["ccd"] = size / 4  # small fast objects (doc 6.2)
        self.objects.append(o)
        return o

    def tint(self):
        import colorsys
        rng = self.rng
        if rng.uniform() < 0.5:
            return [1.0, 1.0, 1.0]
        return list(colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.1, 0.5), 1.0))

    def place(self, o, q_world, xy, z_extra=0.0):
        """Rest an object on the floor at xy with world orientation q_world (applied to the upright frame)."""
        q = shapes.quat_mul(q_world, o["upright"])
        pts = shapes.local_points(o["geom"], self.lib) @ rot(q).T
        o["quat"] = [float(x) for x in q]
        o["pos"] = [float(xy[0]), float(xy[1]), float(-pts[:, 2].min() + 0.0005 + z_extra)]

    def static_box(self, half, pos, quat, tag, texture=None):
        o = self.obj("box", 2 * max(half), role="static", tag=tag, texture=texture)
        o["geom"]["half"] = [float(h) for h in half]
        o.update(pos=[float(x) for x in pos], quat=[float(x) for x in quat], mass_kg=0.0, scale_cm=round(200 * max(half), 2))
        return o

    def pen(self, cx, cy, hx, hy, yaw, height, texture=None):
        texture = texture or self.texture(False)
        t = 0.0125
        R = rot(yaw_quat(yaw))
        for dx, dy, h in ((hx + t, 0, (t, hy + 2 * t, height / 2)), (-hx - t, 0, (t, hy + 2 * t, height / 2)),
                          (0, hy + t, (hx, t, height / 2)), (0, -hy - t, (hx, t, height / 2))):
            p = np.array([cx, cy, 0]) + R @ np.array([dx, dy, height / 2])
            self.static_box(h, p, yaw_quat(yaw), "wall", texture)

    def size(self, lo, hi):
        return loguniform(self.rng, lo, hi)

    def shape(self, allowed):
        """Real meshes take `assets.mesh_fraction` of draws where allowed (doc 6.2: ~40%)."""
        prims = [s for s in allowed if s != "mesh"]
        if "mesh" in allowed and (not prims or self.rng.uniform() < self.cfg["assets"]["mesh_fraction"]):
            return "mesh"
        return prims[self.rng.integers(len(prims))]

    def sep_ok(self, xy, r, placed):
        return all(np.hypot(xy[0] - x, xy[1] - y) > r + rr + 0.01 for x, y, rr in placed)


def bound_radius(o, lib):
    return float(np.linalg.norm(shapes.local_points(o["geom"], lib), axis=1).max())


def enter_frame(rng, t_contact, D):
    """Strikers may roll into view; they are framed (and must be visible) from 0.5-0.8 s before contact."""
    return int(max(0.12 * D["frames"], (t_contact - rng.uniform(0.5, 0.8)) * D["fps"]))


def rolling_omega(v, r):
    return (np.cross([0, 0, 1.0], v) / r).tolist()


# ---------------------------------------------------------------- scenarios
def free_fall(b, D, allowed):
    rng = b.rng
    placed = []
    for _ in range(rng.integers(1, 4)):
        o = b.obj(b.shape(allowed), b.size(0.05, 0.25))
        r = bound_radius(o, b.lib)
        for _ in range(50):
            xy = rng.uniform(-0.25, 0.25, 2)
            if b.sep_ok(xy, r, placed):
                break
        placed.append((*xy, r))
        b.place(o, random_quat(rng), xy, z_extra=rng.uniform(0.1, 0.5))
        # tossed upwards with different speeds so landings spread over the first ~1.2 s
        o["lin_vel"] = [float(rng.uniform(-0.4, 0.4)), float(rng.uniform(-0.4, 0.4)), float(rng.uniform(0.0, 3.2))]
        o["ang_vel"] = rng.uniform(-5, 5, 3).tolist()
    return {}


def ramp_slide(b, D, allowed):
    rng = b.rng
    T = D["frames"]
    theta = np.radians(rng.uniform(18, 35))
    L, W = rng.uniform(0.35, 0.6), rng.uniform(0.2, 0.3)
    shape = b.shape(allowed)
    o = b.obj(shape, b.size(0.06, min(0.14, 0.9 * W)))
    if shape in ("cylinder", "capsule"):  # lying across the slope so it rolls down
        q_can = axis_quat([1, 0, 0], np.pi / 2)
    else:
        q_can = [0, 0, 0, 1]
    q_o = shapes.quat_mul(q_can, o["upright"])
    pts = shapes.local_points(o["geom"], b.lib) @ rot(q_o).T  # in ramp frame
    h_obj = pts[:, 2].max() - pts[:, 2].min()
    h_gate = float(max(0.02, 0.55 * h_obj))
    thick = h_gate + 0.02
    yaw = rng.uniform(0, 2 * np.pi)
    q_ramp = shapes.quat_mul(yaw_quat(yaw), axis_quat([0, 1, 0], -theta))
    R = rot(q_ramp)
    centre = np.array([0, 0, L / 2 * np.sin(theta) - thick / 2 * np.cos(theta) + 0.002])
    ramp = b.static_box([L / 2, W / 2, thick / 2], centre, q_ramp, "ramp")
    mu_obj = o["friction"]
    ramp["friction"] = float(min(1.0, 0.7 * np.tan(theta) / mu_obj) * rng.uniform(0.6, 1.0))
    x_obj = -L / 2 + L * rng.uniform(0.62, 0.85)
    surf = lambda x, z: centre + R @ np.array([x, 0, thick / 2 + z])
    o["quat"] = [float(v) for v in shapes.quat_mul(q_ramp, q_o)]
    o["pos"] = surf(x_obj, -pts[:, 2].min() + 0.0005).tolist()
    gate = b.static_box([0.005, W / 2 * 0.98, h_gate / 2], surf(x_obj + pts[:, 0].min() - 0.0065, h_gate / 2), q_ramp, "gate")
    gate["role"] = "kinematic"
    gate["friction"] = 0.0
    release = int(rng.integers(int(0.3 * T), int(0.8 * T) + 1))
    gate["kinematic"] = dict(start_frame=release, duration_frames=3, delta=(-R[:, 2] * (h_gate + 0.008)).tolist())
    down = -np.array([np.cos(yaw), np.sin(yaw), 0])  # border past the lower end keeps the object in frame
    wall_at = down * (L / 2 * np.cos(theta) + rng.uniform(0.12, 0.3))
    b.static_box([0.0125, W / 2 + 0.15, 0.03], wall_at + [0, 0, 0.03], yaw_quat(yaw), "wall")
    return dict(release_frame=release, cause=[gate["id"]], effect=[o["id"]])


def rolling(b, D, allowed):
    rng = b.rng
    hx, hy = rng.uniform(0.25, 0.4, 2)
    yaw = rng.uniform(0, 2 * np.pi)
    b.pen(0, 0, hx, hy, yaw, rng.uniform(0.03, 0.06))
    R = rot(yaw_quat(yaw))
    placed = []
    for _ in range(rng.integers(1, 4)):
        shape = b.shape(allowed)
        o = b.obj(shape, b.size(0.05, 0.14))
        r = o["geom"]["radius"]
        a = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(a), np.sin(a), 0]) * rng.uniform(0.25, 0.8)
        q = [0, 0, 0, 1] if shape == "sphere" else shapes.quat_mul(yaw_quat(a), axis_quat([1, 0, 0], np.pi / 2))
        for _ in range(50):
            xy = (R @ np.array([rng.uniform(-hx, hx) * 0.7, rng.uniform(-hy, hy) * 0.7, 0]))[:2]
            if b.sep_ok(xy, bound_radius(o, b.lib), placed):
                break
        placed.append((*xy, bound_radius(o, b.lib)))
        b.place(o, q, xy)
        o["lin_vel"], o["ang_vel"] = v.tolist(), rolling_omega(v, r)
    return {}


def billiards(b, D, allowed):
    rng = b.rng
    dur = D["frames"] / D["fps"]
    tmat, _ = b.material()
    while tmat == "metal":
        tmat, _ = b.material()
    tshape = b.shape(allowed)
    target = b.obj(tshape, b.size(0.07, 0.15), tag="target", material=tmat)
    striker = b.obj("sphere", b.size(0.07, 0.15), tag="striker", min_density=3 * target["density"])
    a = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(a), np.sin(a), 0])
    rt, rs = bound_radius(target, b.lib), striker["geom"]["radius"]
    q_t = [0, 0, 0, 1] if tshape != "capsule" else shapes.quat_mul(yaw_quat(a + np.pi / 2), axis_quat([1, 0, 0], np.pi / 2))
    b.place(target, q_t, [0, 0])
    t_c = rng.uniform(0.34, 0.85) * dur  # c3 onset t_c - delta then covers the same window as c1/c2 onsets
    gap = rng.uniform(0.3, 0.7) * t_c
    dist = gap + rs + rt * 0.8
    off = rng.uniform(-0.5, 0.5) * (rs + rt) * np.array([-u[1], u[0], 0])
    start = -u * dist + off
    v = u * gap / t_c
    b.place(striker, [0, 0, 0, 1], start[:2])
    striker["lin_vel"], striker["ang_vel"] = v.tolist(), rolling_omega(v, rs)
    for k in range(rng.integers(0, 3)):  # extra resting targets behind the first one
        e = b.obj(b.shape(allowed), b.size(0.05, 0.1), tag="target")
        pos = u * rng.uniform(0.12, 0.3) + rng.uniform(-0.12, 0.12) * np.array([-u[1], u[0], 0])
        if all(np.hypot(*(pos[:2] - np.array(q["pos"][:2]))) > bound_radius(e, b.lib) + bound_radius(q, b.lib) + 0.01
               for q in b.objects[:-1]):
            b.place(e, [0, 0, 0, 1] if e["shape"] != "capsule" else axis_quat([1, 0, 0], np.pi / 2), pos[:2])
        else:
            b.objects.pop()
    if rng.uniform() < 0.5:
        hx = (dist + 0.5) / 2
        cx = (0.35 - dist) / 2
        b.pen(cx * u[0], cx * u[1], hx, rng.uniform(0.25, 0.35), a, 0.04)
    return dict(cause=[striker["id"]], effect=[target["id"]], enter_frame=enter_frame(rng, t_c, D))


def stack(b, D, allowed):
    rng = b.rng
    dur = D["frames"] / D["fps"]
    yaw = rng.uniform(0, 2 * np.pi)
    z = 0.0
    blocks = []
    for k in range(rng.integers(2, 6)):
        o = b.obj(b.shape(allowed), b.size(0.08, 0.16), tag="block")
        hz = o["geom"]["half"][2]
        o["quat"] = yaw_quat(yaw + rng.uniform(-0.15, 0.15))
        jitter = rng.uniform(-0.1, 0.1, 2) * min(o["geom"]["half"][:2])
        o["pos"] = [float(jitter[0]), float(jitter[1]), float(z + hz + 0.0005)]
        z += 2 * hz + 0.0005
        blocks.append(o)
    striker = b.obj("sphere", b.size(0.08, 0.15), tag="striker", min_density=3 * max(o["density"] for o in blocks))
    a = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(a), np.sin(a), 0])
    t_c = rng.uniform(0.34, 0.85) * dur
    rs = striker["geom"]["radius"]
    gap = rng.uniform(0.5, 0.9) * t_c
    dist = gap + rs + max(blocks[0]["geom"]["half"][:2])
    v = u * gap / t_c
    b.place(striker, [0, 0, 0, 1], (-u * dist)[:2])
    striker["lin_vel"], striker["ang_vel"] = v.tolist(), rolling_omega(v, rs)
    return dict(cause=[striker["id"]], effect=[o["id"] for o in blocks], enter_frame=enter_frame(rng, t_c, D))


def dominoes(b, D, allowed):
    rng = b.rng
    n = int(rng.integers(4, 10))
    h = rng.uniform(0.08, 0.15)
    spacing = rng.uniform(0.5, 0.7) * h
    mat, _ = b.material()
    tex = b.texture(False)
    heading = rng.uniform(0, 2 * np.pi)
    turn = rng.uniform(-0.1, 0.1)
    p = np.zeros(2)
    ids = []
    h0 = heading
    for k in range(n):
        o = b.obj("domino", h * rng.uniform(0.97, 1.03), tag="domino", material=mat, texture=tex)
        b.place(o, yaw_quat(heading), p)
        ids.append(o["id"])
        heading += turn
        p = p + spacing * np.array([np.cos(heading), np.sin(heading)])
    # a rolling ball starts the chain inside the onset window (a pre-tilted domino falls within ~0.3 s)
    dur = D["frames"] / D["fps"]
    striker = b.obj("sphere", 2 * h * rng.uniform(0.3, 0.45), tag="striker", min_density=b.objects[ids[0]]["density"])
    rs = striker["geom"]["radius"]
    u = np.array([np.cos(h0), np.sin(h0), 0.0])
    t_hit = rng.uniform(0.25, 0.65) * dur
    gap = rng.uniform(0.25, 0.5) * t_hit
    b.place(striker, [0, 0, 0, 1], (-u * (gap + rs + b.objects[ids[0]]["geom"]["half"][0]))[:2])
    v = u * gap / t_hit
    striker["lin_vel"], striker["ang_vel"] = v.tolist(), rolling_omega(v, rs)
    centre = np.mean([b.objects[i]["pos"][:2] for i in ids], 0)
    for o in b.objects:
        o["pos"][0] -= float(centre[0])
        o["pos"][1] -= float(centre[1])
    return dict(chain=[striker["id"]] + ids, enter_frame=enter_frame(rng, t_hit, D))


def projectile_bin(b, D, allowed):
    rng = b.rng
    a = rng.uniform(0.12, 0.2)
    yaw = rng.uniform(0, 2 * np.pi)
    b.pen(0, 0, a, a, yaw, rng.uniform(0.08, 0.13))
    g = 9.81
    for _ in range(rng.integers(1, 4)):
        o = b.obj(b.shape(allowed), b.size(0.04, 0.1))
        az = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0.4, 0.65)
        p0 = np.array([r0 * np.cos(az), r0 * np.sin(az), rng.uniform(0.15, 0.4)])
        land = np.r_[rng.uniform(-0.6, 0.6, 2) * a, bound_radius(o, b.lib)]
        tf = rng.uniform(0.35, 0.65)
        v = (land - p0 - 0.5 * np.array([0, 0, -g]) * tf ** 2) / tf
        o["quat"] = random_quat(rng)
        o["pos"] = p0.tolist()
        o["lin_vel"], o["ang_vel"] = v.tolist(), rng.uniform(-5, 5, 3).tolist()
    return {}


BUILDERS = dict(free_fall=free_fall, ramp_slide=ramp_slide, rolling=rolling, billiards=billiards, stack=stack,
                dominoes=dominoes, projectile_bin=projectile_bin)


# ---------------------------------------------------------------- top level
def sample_scene(split, index, cfg, lib, attempt=0):
    seed = scene_seed(split, index)
    pol = policy(split, cfg)
    clip = cfg["dataset"]["clip"]
    res = clip["resolution"]
    res = [res, res] if isinstance(res, int) else list(res)
    hz = cfg["physics"]["hz"]
    if hz % clip["fps"]:
        raise ValueError("physics.hz must be a multiple of the clip fps")
    D = dict(frames=clip["frames"], fps=clip["fps"])
    u = sobol_point(split, index)

    rs = rng_for(seed, "scenario", attempt)
    scenario = choose_scenario(rs, pol, cfg, lib)

    re = rng_for(seed, "env", attempt)
    floors = lib.ids("floor", heldout=pol["assets_heldout"])
    hdris = lib.ids("hdri", heldout=pol["assets_heldout"])
    floor = floors[re.integers(len(floors))]
    albedo = re.uniform(0.2, 0.45)
    env = dict(floor_texture=floor, floor_uv_m=float(re.uniform(0.5, 2.0)),
               floor_gain=float(albedo / max(lib.texture_mean(floor), 1e-3)), floor_contrast=float(re.uniform(0.4, 1.0)),
               floor_friction=float(re.uniform(0.8, 1.0)), floor_restitution=float(re.uniform(0.7, 1.0)),
               floor_roughness=float(re.uniform(0.4, 0.9)), hdri=hdris[re.integers(len(hdris))],
               hdri_rot_deg=float(re.uniform(0, 360)), ambient=float(re.uniform(*cfg["light"]["ambient"])),
               exposure_ev=float(re.uniform(*cfg["renderer"]["exposure_ev"])))

    b = Builder(rng_for(seed, "objects", attempt), lib, pol, cfg)
    info = BUILDERS[scenario](b, D, allowed_shapes(scenario, pol, cfg, lib))

    lc = cfg["light"]
    light = dict(azimuth=float(360 * u[0]), elevation=float(lc["elevation_deg"][0] + u[1] * np.ptp(lc["elevation_deg"])),
                 intensity=float(rng_for(seed, "light", attempt).uniform(2.0, 5.0)),
                 color_temp=float(lc["color_temp_k"][0] + u[5] * np.ptp(lc["color_temp_k"])),
                 softness=float(lc["softness"][0] + u[6] * np.ptp(lc["softness"])))
    return dict(scene_seed=seed, scene_index=index, attempt=attempt, split=split, generator_version=__version__,
                generator_commit=__commit__,
                asset_manifest=lib.manifest_hash, scenario=scenario, info=info, sobol=u.tolist(),
                clip=dict(frames=clip["frames"], fps=clip["fps"], substeps=hz // clip["fps"], resolution=res),
                physics=dict(engine=cfg["physics"]["engine"], hz=hz, gravity=cfg["physics"]["gravity"],
                             solver_iterations=cfg["physics"]["solver_iterations"],
                             internal_substeps=cfg["physics"]["internal_substeps"]),
                environment=env, light=light, objects=b.objects)


def add_distractors(scene, cam, footprint, cfg, lib):
    """0-5 static distractors in view, never overlapping the action footprint (doc 6.5)."""
    from . import camera
    rng = rng_for(scene["scene_seed"], "clutter", scene["attempt"])
    pol = policy(scene["split"], cfg)
    b = Builder(rng, lib, pol, cfg)
    b.objects = scene["objects"]
    lo, hi = cfg["clutter"]["static_distractors"]
    want = int(rng.integers(lo, hi + 1))
    eye, _ = camera.pose(cam, 0.0)
    V = camera.view_matrix(*camera.pose(cam, 0.0))
    f = 1.0 / np.tan(np.radians(cam["fov_deg"]) / 2)
    placed = [(x, y, r) for x, y, r in footprint]
    use_mesh = "ycb" in cfg["assets"]["meshes"] and lib.ids("mesh", heldout=pol["meshes_heldout"])
    shapes_ = [s for s in ["box", "cylinder", "sphere", "wedge", "hull", "brick"] + (["mesh"] if use_mesh else [])
               if (pol["shapes"] is None and s not in pol["heldout_shapes"]) or (pol["shapes"] and s in pol["shapes"])]
    shapes_ = shapes_ or ["box"]
    for _ in range(want):
        o = b.obj(b.shape(shapes_), b.size(0.04, 0.15), role="static", tag="distractor")
        r = bound_radius(o, lib)
        for _ in range(40):
            nx, ny = rng.uniform(-0.9, 0.9, 2)
            d = -V[2, :3] + V[0, :3] * nx / f + V[1, :3] * ny / f
            if d[2] > -1e-3:
                continue
            hit = eye + d * (-eye[2] / d[2])
            if np.linalg.norm(hit - eye) < 4.0 and b.sep_ok(hit[:2], r, placed):
                b.place(o, yaw_quat(rng.uniform(0, 2 * np.pi)), hit[:2])
                o["mass_kg"] = 0.0
                placed.append((hit[0], hit[1], r))
                break
        else:
            b.objects.pop()
