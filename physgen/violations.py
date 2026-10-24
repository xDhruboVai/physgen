"""Violation injectors (doc 6.3, 8.2 'injectors').

c1 edits the light track (shadow light direction/softness and shadow-caster poses), keeping the c0
trajectories; c2 edits the simulation through backend operations and lets it continue; c3
time-advances the effect trajectory by delta frames.

Each builder returns a Violation or None when the scene cannot host that subtype.
Frames are half-open intervals [onset, offset).
"""
import numpy as np

from . import shapes
from .physics import FLOOR

SUBTYPES = {
    "S1.1": (1, "shadow_direction_mismatch", ("persistent", "transient")),
    "S1.2": (1, "detached_shadow", ("persistent", "transient")),
    "S1.3": (1, "frozen_shadow", ("persistent", "transient")),
    "S1.4": (1, "shadow_elevation_mismatch", ("persistent", "transient")),
    "S1.5": (1, "reversed_shadow_motion", ("persistent", "transient")),
    "S1.6": (1, "softness_mismatch", ("persistent", "transient")),
    "S2.1": (2, "phase_through", ("event",)),
    "S2.2": (2, "spontaneous_impulse", ("transient",)),
    "S2.3": (2, "gravity_change", ("persistent", "transient")),
    "S2.4": (2, "mid_air_stop", ("persistent", "transient")),
    "S2.5": (2, "super_elastic_bounce", ("persistent", "transient")),
    "S2.6": (2, "mass_inconsistency", ("transient",)),
    "S2.7": (2, "torque_free_spin", ("persistent", "transient")),
    "S2.8": (2, "friction_anomaly", ("persistent", "transient")),
    "S3.1": (3, "early_reaction", ("event",)),
    "S3.2": (3, "early_domino_fall", ("event",)),
    "S3.3": (3, "early_toppling", ("event",)),
    "S3.4": (3, "early_release", ("event",)),
}

# severity -> magnitude range per subtype (starting values, tuned with the visibility validator V6)
SEVERITY = {
    "S1.1": dict(subtle=(30, 60), clear=(60, 110), strong=(110, 180)),          # deg azimuth offset
    "S1.2": dict(subtle=(0.05, 0.1), clear=(0.1, 0.25), strong=(0.25, 0.5)),    # m/s drift
    "S1.3": dict(subtle=(0.3, 0.5), clear=(0.6, 0.85), strong=(1.0, 1.0)),      # freeze fraction
    "S1.4": dict(subtle=(8, 15), clear=(15, 25), strong=(25, 40)),              # deg elevation offset
    "S1.5": dict(subtle=(0.3, 0.6), clear=(0.6, 1.0), strong=(1.0, 1.5)),       # reversal gain
    "S1.6": dict(subtle=(1, 1), clear=(2, 2), strong=(2, 2)),                   # penumbra mode
    "S2.1": dict(subtle=(0, 0.6), clear=(0.6, 1.5), strong=(1.5, 99)),          # contact speed (m/s), post hoc
    "S2.2": dict(subtle=(0.25, 0.5), clear=(0.5, 1.0), strong=(1.0, 2.0)),      # delta-v m/s
    "S2.3": dict(subtle=(0.4, 0.7), clear=(0.0, 0.25), strong=(-1.0, -0.4)),    # gravity factor
    "S2.4": dict(subtle=(3, 6), clear=(6, 12), strong=(12, 24)),                # stop window frames
    "S2.5": dict(subtle=(1.05, 1.25), clear=(1.25, 1.6), strong=(1.6, 2.2)),    # effective restitution
    "S2.6": dict(subtle=(2, 3), clear=(4, 8), strong=(10, 20)),                 # inverted mass ratio
    "S2.7": dict(subtle=(3, 6), clear=(6, 12), strong=(12, 25)),                # delta-omega rad/s
    "S2.8": dict(subtle=(1.0, 1.5), clear=(2.0, 3.0), strong=(4.0, 6.0)),       # stop factor / boost m/s^2
    "S3.1": dict(subtle=(4, 6), clear=(7, 10), strong=(11, 14)),                # delta frames
    "S3.2": dict(subtle=(4, 6), clear=(7, 10), strong=(11, 14)),
    "S3.3": dict(subtle=(4, 6), clear=(7, 10), strong=(11, 14)),
    "S3.4": dict(subtle=(4, 6), clear=(7, 10), strong=(11, 14)),
}


class Violation:
    def __init__(self, subtype, severity, timing, onset, involved, params):
        self.subtype, self.cls = subtype, SUBTYPES[subtype][0]
        self.severity, self.timing = severity, timing
        self.inject_frame = onset          # frame at which the change is applied
        self.offset = None                 # first frame after the violation (set by the builder or after sim)
        self.involved = [int(i) for i in involved]
        self.params = dict(type=SUBTYPES[subtype][1], **params)
        self.hooks = []                    # c2
        self.track = None                  # c1: light track
        self.traj = None                   # c3: modified trajectory

    def meta(self):
        return dict(subtype=self.subtype, severity=self.severity, timing=self.timing,
                    inject_frame=int(self.inject_frame), involved_objects=self.involved,
                    violation_params=_jsonable(self.params))


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---------------------------------------------------------------- light track (c1 and the c0 default)
def light_dir(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def base_track(scene, tr):
    T = scene["clip"]["frames"]
    L = scene["light"]
    d = light_dir(L["azimuth"], L["elevation"])
    return dict(shade_dir=np.tile(d, (T, 1)), shadow_dir=np.tile(d, (T, 1)),
                softness=np.full(T, softness_rad(L["softness"])), mode=np.zeros(T, np.int32),
                spos=tr.pos.copy(), squat=tr.quat.copy())


def softness_rad(s):
    """Softness in [0,1] -> angular radius of the key light (0.25 to 5 degrees)."""
    return np.radians(0.25 + 4.75 * s)


def ramp_profile(T, on, timing, w, ramp=4):
    """0 -> 1 over `ramp` frames from `on`; transient: hold w frames then back to 0. Returns (profile, offset)."""
    f = np.arange(T, dtype=float)
    up = np.clip((f - on + 1) / ramp, 0, 1)
    if timing == "persistent":
        return up, T
    end = on + ramp + w
    down = np.clip(1 - (f - end + 1) / ramp, 0, 1)
    return np.minimum(up, down), min(T, end + ramp - 1)


# ---------------------------------------------------------------- helpers over the c0 trajectory
class Ctx:
    def __init__(self, scene, tr, rng, cfg):
        self.scene, self.tr, self.rng, self.cfg = scene, tr, rng, cfg
        self.T = scene["clip"]["frames"]
        self.S = scene["clip"]["substeps"]
        self.fps = scene["clip"]["fps"]
        lo, hi = cfg["violations"]["onset_range"]
        self.lo, self.hi = int(round(lo * self.T)), int(round(hi * self.T))
        self.objs = scene["objects"]
        self.dyn = [o["id"] for o in self.objs if o["role"] == "dynamic"]
        self.static = {FLOOR} | {o["id"] for o in self.objs if o["role"] != "dynamic"}

    def onset(self):
        return int(self.rng.integers(self.lo, self.hi + 1))

    def window(self, on, lo=8, hi=24, tail=4):
        hi = min(hi, self.T - tail - on)
        if hi < lo:
            return None
        return int(self.rng.integers(lo, hi + 1))

    def pos(self, i, f):
        return self.tr.pos[i, f * self.S]

    def speed(self, i, f):
        return float(np.linalg.norm(self.tr.lin[i, f * self.S]))

    def in_contact(self, i, f, with_static_only=False):
        for a, b in self.tr.contacts[f * self.S]:
            if i in (a, b):
                other = b if a == i else a
                if not with_static_only or other in self.static:
                    return True
        return False

    def displacement(self, i, f0, f1, horizontal=False):
        p = self.tr.pos[i, f0 * self.S:f1 * self.S + 1:self.S]
        if horizontal:
            p = p[:, :2]
        return float(np.linalg.norm(p - p[0], axis=1).max()) if len(p) else 0.0

    def pick(self, items, weights=None):
        if not items:
            return None
        w = None if weights is None else np.asarray(weights, float) / np.sum(weights)
        return items[self.rng.choice(len(items), p=w)]

    def events_in_range(self, pred, lo=None, hi=None):
        lo = self.lo if lo is None else lo
        hi = self.hi if hi is None else hi
        out = []
        for e in self.tr.events:
            f = e["step"] // self.S
            if lo <= f <= hi and pred(e):
                out.append(e)
        return out


def magnitude(ctx, subtype, severity):
    a, b = SEVERITY[subtype][severity]
    return float(ctx.rng.uniform(a, b))


def build(subtype, severity, timing, scene, tr, rng, cfg):
    ctx = Ctx(scene, tr, rng, cfg)
    return BUILDERS[subtype](ctx, subtype, severity, timing)


# ---------------------------------------------------------------- c1
def _c1_dir(ctx, st, sev, timing, which):
    on = ctx.onset()
    w = ctx.window(on, tail=12) if timing == "transient" else 0
    if w is None:
        return None
    L = ctx.scene["light"]
    mag = magnitude(ctx, st, sev) * ctx.rng.choice([-1, 1])
    if which == "el":
        el = L["elevation"]
        if not 12 <= el + mag <= 85:
            mag = -mag
        mag = float(np.clip(el + mag, 12, 85) - el)
    prof, off = ramp_profile(ctx.T, on, timing, w)
    v = Violation(st, sev, timing, on, [], dict(magnitude=mag, axis=which, window=w))
    tk = base_track(ctx.scene, ctx.tr)
    for f in range(ctx.T):
        az = L["azimuth"] + (mag * prof[f] if which == "az" else 0)
        el = L["elevation"] + (mag * prof[f] if which == "el" else 0)
        tk["shadow_dir"][f] = light_dir(az, el)
    v.track, v.offset = tk, off
    return v


def s1_1(ctx, st, sev, timing):
    return _c1_dir(ctx, st, sev, timing, "az")


def s1_4(ctx, st, sev, timing):
    return _c1_dir(ctx, st, sev, timing, "el")


def _c1_objects(ctx, need_motion, horizontal=False):
    """One moving object, or all dynamic objects (30%)."""
    on = ctx.onset()
    cands = [i for i in ctx.dyn if not need_motion or ctx.displacement(i, on, ctx.T - 1, horizontal) > 0.05]
    if not cands:
        return on, None
    if ctx.rng.uniform() < 0.3 and not need_motion:
        return on, cands
    return on, [ctx.pick(cands)]


def _step_profile(ctx, on, timing, w, ramp=4):
    prof, off = ramp_profile(ctx.T + 1, on, timing, w, ramp)
    k = np.arange(ctx.tr.pos.shape[1]) / ctx.S
    return np.interp(k, np.arange(ctx.T + 1), prof), min(off, ctx.T)


def s1_2(ctx, st, sev, timing):
    on, ids = _c1_objects(ctx, need_motion=False)
    if ids is None:
        return None
    w = ctx.window(on) if timing == "transient" else 0
    if w is None:
        return None
    rate = magnitude(ctx, st, sev)
    a = ctx.rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(a), np.sin(a), 0])
    t = (np.arange(ctx.tr.pos.shape[1]) / ctx.S - on) / ctx.fps
    if timing == "persistent":
        amount, off = np.clip(t, 0, None) * rate, ctx.T
    else:
        half = w / 2 / ctx.fps
        amount, off = np.clip(np.minimum(t, w / ctx.fps - t), 0, half) * rate, min(ctx.T, on + w)
    v = Violation(st, sev, timing, on, ids, dict(magnitude=rate, direction_deg=np.degrees(a), window=w))
    tk = base_track(ctx.scene, ctx.tr)
    for i in ids:
        tk["spos"][i] += amount[:, None] * d
    v.track, v.offset = tk, off
    return v


def s1_3(ctx, st, sev, timing):
    on, ids = _c1_objects(ctx, need_motion=True)
    if ids is None:
        return None
    w = ctx.window(on, tail=12) if timing == "transient" else 0
    if w is None:
        return None
    alpha = magnitude(ctx, st, sev)
    prof, off = _step_profile(ctx, on, timing, w)
    prof = prof * alpha
    from scipy.spatial.transform import Rotation, Slerp
    v = Violation(st, sev, timing, on, ids, dict(magnitude=alpha, window=w))
    tk = base_track(ctx.scene, ctx.tr)
    k0 = on * ctx.S
    for i in ids:
        frozen_p, frozen_q = ctx.tr.pos[i, k0], ctx.tr.quat[i, k0]
        tk["spos"][i] = ctx.tr.pos[i] + prof[:, None] * (frozen_p - ctx.tr.pos[i])
        for k in np.nonzero(prof > 0)[0]:
            sl = Slerp([0, 1], Rotation.from_quat([ctx.tr.quat[i, k], frozen_q]))
            tk["squat"][i, k] = sl(prof[k]).as_quat()
    v.track, v.offset = tk, off
    return v


def s1_5(ctx, st, sev, timing):
    on, ids = _c1_objects(ctx, need_motion=True, horizontal=True)
    if ids is None:
        return None
    w = ctx.window(on, tail=16) if timing == "transient" else 0
    if w is None:
        return None
    beta = magnitude(ctx, st, sev)
    prof, off = _step_profile(ctx, on, timing, w, ramp=6)
    v = Violation(st, sev, timing, on, ids, dict(magnitude=beta, window=w))
    tk = base_track(ctx.scene, ctx.tr)
    k0 = on * ctx.S
    for i in ids:
        p = ctx.tr.pos[i]
        mirrored = p.copy()
        mirrored[:, :2] = p[k0, :2] - beta * (p[:, :2] - p[k0, :2])
        tk["spos"][i] = p + prof[:, None] * (mirrored - p) * (np.arange(len(p)) >= k0)[:, None]
    v.track, v.offset = tk, off
    return v


def s1_6(ctx, st, sev, timing):
    if ctx.scene["light"]["softness"] < 0.25:
        return None
    on = ctx.onset()
    # needs occluders at varying heights: vertical motion after onset, or tall objects (stacks, ramps)
    zr = [np.ptp(ctx.tr.pos[i, on * ctx.S:ctx.T * ctx.S:ctx.S, 2]) + ctx.tr.pos[i, on * ctx.S, 2] for i in ctx.dyn]
    if not zr or max(zr) < 0.08:
        return None
    w = ctx.window(on) if timing == "transient" else 0
    if w is None:
        return None
    mode = int(SEVERITY[st][sev][0])
    v = Violation(st, sev, timing, on, [ctx.dyn[int(np.argmax(zr))]], dict(mode=["normal", "constant", "inverted"][mode],
                                                                           softness_scale=1.5 if sev == "strong" else 1.0, window=w))
    tk = base_track(ctx.scene, ctx.tr)
    end = ctx.T if timing == "persistent" else on + w
    tk["mode"][on:end] = mode
    tk["softness"][on:end] *= v.params["softness_scale"]
    tk["mode_obj"] = v.involved[0]  # the Blender backend approximates S1.6 from this object's height
    v.track, v.offset = tk, end
    return v


# ---------------------------------------------------------------- c2 (simulation edits)
class Hook:
    """Called before every physics step (k) and, via after(), after it."""

    def __init__(self, k_on):
        self.k_on = k_on

    def __call__(self, sim, k, tr):
        pass

    def after(self, sim, k, tr):
        pass


def s2_1(ctx, st, sev, timing):
    evs = ctx.events_in_range(lambda e: (e["a"] in ctx.dyn or e["b"] in ctx.dyn) and e["speed"] > 0.2)
    if not evs:
        return None
    e = ctx.pick(evs, [1 if FLOOR in (e["a"], e["b"]) else 3 for e in evs])
    i, j = (e["a"], e["b"]) if e["a"] in ctx.dyn else (e["b"], e["a"])
    sev = next(s for s, (lo, hi) in SEVERITY[st].items() if lo <= e["speed"] < hi)
    k_on = max(e["step"] - 2 * ctx.S, ctx.lo * ctx.S)  # contact reports can lag the impulse by a step

    class H(Hook):
        overlapped = False

        def __call__(self, sim, k, tr):
            if k == self.k_on:
                sim.set_collision(i, j, False)

        def after(self, sim, k, tr):
            if k >= self.k_on:
                tr.overlap[k + 1] = sim.overlapping(i, j)
                if tr.overlap[k + 1]:
                    self.overlapped = True
                elif self.overlapped and FLOOR not in (i, j):
                    sim.set_collision(i, j, True)  # separated again: the pass-through is over

    v = Violation(st, sev, timing, k_on // ctx.S, [i] + ([j] if j != FLOOR else []),
                  dict(pair=[i, j], contact_speed=e["speed"]))
    v.hooks = [H(k_on)]
    v.offset_from_sim = lambda tr: _overlap_interval(tr, ctx)
    return v


def _overlap_interval(tr, ctx):
    f = np.nonzero(tr.overlap[::ctx.S][:ctx.T])[0]
    return None if not len(f) else (int(f[0]), int(f[-1]) + 1)


def _force_hook(i, k_on, k_off, force_fn):
    class H(Hook):
        def __call__(self, sim, k, tr):
            if self.k_on <= k < k_off:
                sim.apply_force(i, force_fn(sim, k))
    return H(k_on)


def s2_2(ctx, st, sev, timing):
    """Instantaneous velocity kick with no contact; the motion afterwards is ordinary physics.
    (A force spread over frames is mostly absorbed by static friction for resting objects.)"""
    on = ctx.onset()
    i = ctx.pick(ctx.dyn)
    w = 3  # labelled frames: the kick and its immediate effect
    if on + w > ctx.T - 4:
        return None
    dv = magnitude(ctx, st, sev)
    a = ctx.rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(a), np.sin(a), ctx.rng.uniform(0.2, 1.2)])
    d /= np.linalg.norm(d)

    class H(Hook):
        def __call__(self, sim, k, tr):
            if k == self.k_on:
                sim.set_velocity(i, sim.state(i)[2] + dv * d)

    v = Violation(st, sev, timing, on, [i], dict(delta_v=dv, direction=d, window=w))
    v.hooks = [H(on * ctx.S)]
    v.offset = on + w
    return v


def s2_3(ctx, st, sev, timing):
    on = ctx.onset()
    w = ctx.window(on) if timing == "transient" else 0
    if w is None:
        return None
    ids = list(ctx.dyn) if ctx.rng.uniform() < 0.3 else [ctx.pick(ctx.dyn)]
    kf = magnitude(ctx, st, sev)
    g = np.array([0, 0, ctx.scene["physics"]["gravity"]])
    end = ctx.T if timing == "persistent" else on + w
    v = Violation(st, sev, timing, on, ids, dict(gravity_factor=kf, scope="scene" if len(ids) > 1 else "object", window=w))
    stop_k = end * ctx.S if timing == "transient" else 10 ** 9
    v.hooks = [_force_hook(i, on * ctx.S, stop_k, lambda sim, k, i=i: ctx.objs[i]["mass_kg"] * (kf - 1) * g)
               for i in ids]
    v.offset = end
    return v


def s2_4(ctx, st, sev, timing):
    cands = [(i, f) for f in range(ctx.lo, ctx.hi + 1) for i in ctx.dyn
             if not ctx.in_contact(i, f) and ctx.speed(i, f) > 0.25 and ctx.pos(i, f)[2] > 0.04]
    if not cands:
        return None
    i, on = cands[ctx.rng.integers(len(cands))]
    w = int(round(magnitude(ctx, st, sev)))
    if timing == "transient" and on + w > ctx.T - 4:
        return None
    end = ctx.T if timing == "persistent" else on + w
    m = ctx.objs[i]["mass_kg"]
    g = np.array([0, 0, ctx.scene["physics"]["gravity"]])
    k_on = on * ctx.S

    class H(Hook):
        def __call__(self, sim, k, tr):
            if k == self.k_on:
                sim.set_velocity(i, np.zeros(3), np.zeros(3))
            if self.k_on <= k < end * ctx.S + (16 * ctx.S if timing == "persistent" else 0):
                sim.apply_force(i, -m * g)

    v = Violation(st, sev, timing, on, [i], dict(window=w))
    v.hooks = [H(k_on)]
    v.offset = end
    return v


def s2_5(ctx, st, sev, timing):
    evs = ctx.events_in_range(lambda e: ((e["a"] in ctx.dyn and e["b"] in ctx.static) or
                                         (e["b"] in ctx.dyn and e["a"] in ctx.static)) and e["speed"] > 0.5)
    if not evs:
        return None
    e = ctx.pick(evs)
    i = e["a"] if e["a"] in ctx.dyn else e["b"]
    e_eff = magnitude(ctx, st, sev)
    k_on = max(e["step"] - 2 * ctx.S, 0)
    bounces = []

    class H(Hook):
        v_prev = None

        def __call__(self, sim, k, tr):
            if k >= self.k_on:
                self.v_prev = sim.state(i)[2]

        def after(self, sim, k, tr):
            if k < self.k_on or (timing == "transient" and bounces):
                return
            new = tr.contacts[k + 1] - tr.contacts[k]
            if not any(i in p and (p[0] if p[1] == i else p[1]) in ctx.static for p in new):
                return
            for a, b, n, d, f in sim.contact_points():
                if i in (a, b) and (b if a == i else a) in ctx.static:
                    n = n if a == i else -n
                    vin = -float(self.v_prev @ n)
                    if vin < 0.3:
                        return
                    lin = sim.state(i)[2]
                    sim.set_velocity(i, lin + (e_eff * vin - float(lin @ n)) * n)
                    bounces.append(k + 1)
                    return

    v = Violation(st, sev, timing, k_on // ctx.S, [i], dict(restitution=e_eff))
    v.hooks = [H(k_on)]

    def interval(tr):
        if not bounces:
            return None
        f0 = bounces[0] // ctx.S
        if timing == "persistent":
            return f0, ctx.T
        vz = abs(tr.lin[i, min(bounces[0] + 1, tr.lin.shape[1] - 1), 2])
        return f0, min(ctx.T, f0 + max(4, int(round(vz / 9.81 * ctx.fps))))

    v.offset_from_sim = interval
    v.params["bounce_steps"] = bounces
    return v


def s2_6(ctx, st, sev, timing):
    evs = ctx.events_in_range(lambda e: e["a"] in ctx.dyn and e["b"] in ctx.dyn and e["speed"] > 0.15)
    if not evs:
        return None
    e = ctx.pick(evs)
    i, j = e["a"], e["b"]
    mi, mj = ctx.objs[i]["mass_kg"], ctx.objs[j]["mass_kg"]
    heavy, light = (i, j) if mi >= mj else (j, i)
    R = magnitude(ctx, st, sev)
    M = mi + mj
    new = {heavy: M / (1 + R), light: M * R / (1 + R)}
    f_c = e["step"] // ctx.S
    k_on = max(e["step"] - 2 * ctx.S, 0)
    k_off = (f_c + 4) * ctx.S
    if f_c + 4 > ctx.T - 4:
        return None

    class H(Hook):
        def __call__(self, sim, k, tr):
            if k == self.k_on:
                self.orig = {x: sim.mass(x) for x in (i, j)}
                for x, m in new.items():
                    sim.set_mass(x, m)
            if k == k_off:
                for x, m in self.orig.items():
                    sim.set_mass(x, m)

    v = Violation(st, sev, timing, k_on // ctx.S, [i, j], dict(pair=[i, j], mass_ratio=R, masses=[new[i], new[j]]))
    v.hooks = [H(k_on)]
    v.offset = f_c + 4
    return v


def s2_7(ctx, st, sev, timing):
    on = ctx.onset()
    i = ctx.pick(ctx.dyn)
    dw = magnitude(ctx, st, sev)
    resting = ctx.in_contact(i, on, with_static_only=True)
    axis = ctx.rng.normal(size=3) * (0.3 if resting else 1.0) + ([0, 0, 1] if resting else 0)
    axis /= np.linalg.norm(axis)
    if timing == "transient":
        w = int(ctx.rng.integers(2, 6))
        if on + w > ctx.T - 4:
            return None
        end, per_step = on + w, dw / (w * ctx.S)
    else:
        w, end, per_step = 0, ctx.T, dw / (0.5 * ctx.fps * ctx.S)
    stop_k = end * ctx.S + (16 * ctx.S if timing == "persistent" else 0)

    class H(Hook):
        def __call__(self, sim, k, tr):
            if self.k_on <= k < stop_k:
                sim.set_velocity(i, ang=sim.state(i)[3] + per_step * axis)

    v = Violation(st, sev, timing, on, [i], dict(delta_omega=dw, axis=axis, window=w))
    v.hooks = [H(on * ctx.S)]
    v.offset = end
    return v


def s2_8(ctx, st, sev, timing):
    cands = []
    for f in range(ctx.lo, ctx.hi + 1):
        for i in ctx.dyn:
            if ctx.objs[i]["shape"] in shapes.ROUND or not ctx.in_contact(i, f, with_static_only=True):
                continue
            if np.linalg.norm(ctx.tr.lin[i, f * ctx.S, :2]) > 0.15:
                cands.append((i, f))
    if not cands:
        return None
    i, on = cands[ctx.rng.integers(len(cands))]
    w = ctx.window(on) if timing == "transient" else 0
    if w is None:
        return None
    end = ctx.T if timing == "persistent" else on + w
    stop_k = end * ctx.S + (16 * ctx.S if timing == "persistent" else 0)
    mode = ctx.pick(["stop", "boost"])
    mag = magnitude(ctx, st, sev)
    m = ctx.objs[i]["mass_kg"]

    class H(Hook):
        def __call__(self, sim, k, tr):
            if k == self.k_on:
                self.mu = sim.friction(i)
                if mode == "stop":
                    sim.set_friction(i, self.mu * (1 + 3 * mag))
                    if sev == "strong":
                        sim.set_velocity(i, np.zeros(3), np.zeros(3))
                else:
                    sim.set_friction(i, 0.0)
            if mode == "boost" and self.k_on <= k < stop_k:
                vel = sim.state(i)[2]
                h = np.array([vel[0], vel[1], 0.0])
                if np.linalg.norm(h) > 1e-3:
                    sim.apply_force(i, m * mag * h / np.linalg.norm(h))
            if k == stop_k:
                sim.set_friction(i, self.mu)

    v = Violation(st, sev, timing, on, [i], dict(mode=mode, magnitude=mag, window=w))
    v.hooks = [H(on * ctx.S)]
    v.offset = end
    return v


# ---------------------------------------------------------------- c3 (time-advance the effect)
def _c3(ctx, st, sev, timing, cause, effect, t_c):
    delta = int(round(magnitude(ctx, st, sev)))
    on = t_c - delta
    if not (ctx.lo <= on <= ctx.hi) or t_c + 1 > ctx.T - 2:
        return None
    S = ctx.S
    # downstream closure: objects the effect set touches after t_c also move early
    eff = set(effect)
    for k in range(t_c * S, min(len(ctx.tr.contacts), (ctx.T + delta) * S)):
        for a, b in ctx.tr.contacts[k]:
            if a in eff and b in ctx.dyn and b not in cause:
                eff.add(b)
            if b in eff and a in ctx.dyn and a not in cause:
                eff.add(a)
    for i in eff:  # the effect must be at rest while it is advanced over [t_c - delta, t_c]
        if np.abs(ctx.tr.lin[i, (on - 1) * S:(t_c - 1) * S + 1]).max() > 0.03:
            return None
    n_steps = ctx.T * S
    if ctx.tr.pos.shape[1] < n_steps + delta * S + 1:
        return None
    pos, quat = ctx.tr.pos.copy(), ctx.tr.quat.copy()
    k0 = (on - 1) * S  # the impulse lands somewhere inside frame t_c - 1; splice while still at rest
    for i in eff:
        pos[i, k0:n_steps + 1] = ctx.tr.pos[i, k0 + delta * S:n_steps + 1 + delta * S]
        quat[i, k0:n_steps + 1] = ctx.tr.quat[i, k0 + delta * S:n_steps + 1 + delta * S]
    v = Violation(st, sev, timing, on, sorted(set(cause) | eff), dict(delta_frames=delta, contact_frame=t_c,
                                                                      cause=cause, effect=sorted(eff)))
    v.traj = (pos, quat)
    v.offset = t_c + 1
    return v


def _first_contact(ctx, A, B):
    """Frame at which the effect B first moves because of A (contact reports can lag the impulse)."""
    for e in ctx.tr.events:
        if (e["a"] in A and e["b"] in B) or (e["b"] in A and e["a"] in B):
            f0 = e["step"] // ctx.S
            for f in range(max(1, f0 - 3), f0 + 2):
                if any(np.linalg.norm(ctx.tr.lin[i, f * ctx.S]) > 0.02 for i in B):
                    return f
            return f0
    return None


def s3_1(ctx, st, sev, timing):
    info = ctx.scene["info"]
    t_c = _first_contact(ctx, info["cause"], info["effect"])
    return None if t_c is None else _c3(ctx, st, sev, timing, info["cause"], info["effect"], t_c)


s3_3 = s3_1


def s3_2(ctx, st, sev, timing):
    chain = ctx.scene["info"]["chain"]
    opts = []
    for k in range(len(chain) - 1):
        t_c = _first_contact(ctx, [chain[k]], [chain[k + 1]])
        if t_c is not None:
            opts.append((k, t_c))
    ctx.rng.shuffle(opts)
    for k, t_c in opts:
        v = _c3(ctx, st, sev, timing, [chain[k]], chain[k + 1:], t_c)
        if v is not None:
            return v
    return None


def s3_4(ctx, st, sev, timing):
    info = ctx.scene["info"]
    return _c3(ctx, st, sev, timing, info["cause"], info["effect"], info["release_frame"])


BUILDERS = {"S1.1": s1_1, "S1.2": s1_2, "S1.3": s1_3, "S1.4": s1_4, "S1.5": s1_5, "S1.6": s1_6,
            "S2.1": s2_1, "S2.2": s2_2, "S2.3": s2_3, "S2.4": s2_4, "S2.5": s2_5, "S2.6": s2_6,
            "S2.7": s2_7, "S2.8": s2_8, "S3.1": s3_1, "S3.2": s3_2, "S3.3": s3_3, "S3.4": s3_4}
