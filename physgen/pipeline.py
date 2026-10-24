"""One scene end to end (doc 8.1): sampler -> sim -> injectors -> renderer -> validators -> clips."""
import numpy as np

from . import camera, physics, sampler, shapes, validators, violations
from .render import common

TAIL = 16  # extra c0 frames simulated so c3 can time-advance the effect trajectory


class Rejected(Exception):
    pass


def action_points(scene, tr, lib, T):
    """Points the camera must frame. Strikers may roll in from outside the view until 0.5-0.8 s
    before contact, and walls may be cut by the frame edge, so objects stay a readable size."""
    S = scene["clip"]["substeps"]
    pts = []
    for o in scene["objects"]:
        local = shapes.local_points(o["geom"], lib)
        if o["role"] == "dynamic":
            r = float(np.linalg.norm(local, axis=1).max())
            f0 = scene["info"].get("enter_frame", int(0.12 * T)) if o["tag"] == "striker" else 0
            c = tr.pos[o["id"], f0 * S:T * S + 1:S]
            pts.append((c[:, None, :] + r * np.vstack([np.eye(3), -np.eye(3)])[None]).reshape(-1, 3))
        elif o["tag"] in ("ramp", "gate"):
            pts.append(local @ shapes.quat_to_mat(o["quat"]).T + o["pos"])
    pts = np.concatenate(pts)
    pts[:, 2] = np.maximum(pts[:, 2], 0.0)
    return pts


def footprint(scene, tr, lib, T):
    """(x, y, r) discs that distractors must avoid: every dynamic position and every static part."""
    S = scene["clip"]["substeps"]
    out = []
    for o in scene["objects"]:
        r = float(np.linalg.norm(shapes.local_points(o["geom"], lib), axis=1).max())
        if o["role"] == "dynamic":
            out += [(x, y, r + 0.05) for x, y in tr.pos[o["id"], :T * S + 1:S * 2, :2]]
        else:
            out.append((o["pos"][0], o["pos"][1], r + 0.05))
    return out


def shadow_bounds(scene, tr, lib, T):
    pts = action_points(scene, tr, lib, T)
    stat = [np.asarray(o["pos"]) for o in scene["objects"] if o["tag"] == "distractor"]
    allp = np.vstack([pts] + ([np.array(stat)] if stat else []))
    c = (allp.min(0) + allp.max(0)) / 2
    c[2] = 0.0
    el = max(scene["light"]["elevation"] - 40, 12)  # c1 may lower the shadow light
    R = np.linalg.norm(allp - c, axis=1).max() + allp[:, 2].max() / np.tan(np.radians(el)) + 0.3
    return c, float(min(R, 4.0))


def simulate_c0(split, index, cfg, lib, log):
    T = cfg["dataset"]["clip"]["frames"]
    for attempt in range(cfg["validation"].get("scene_attempts", 10)):
        scene = sampler.sample_scene(split, index, cfg, lib, attempt)
        tr = physics.simulate(scene, T + TAIL, mesh_lib=lib)
        why = validators.v1_stable(tr) or validators.v2_penetration(tr)
        if why is None:
            W, H = scene["clip"]["resolution"]
            cam_rng = sampler.rng_for(scene["scene_seed"], "camera", attempt)
            scene["camera"] = camera.frame_points(action_points(scene, tr, lib, T), cam_rng, cfg, W / H)
            n0 = len(scene["objects"])
            sampler.add_distractors(scene, scene["camera"], footprint(scene, tr, lib, T), cfg, lib)
            if len(scene["objects"]) > n0:
                tr = physics.simulate(scene, T + TAIL, mesh_lib=lib)
            why = (validators.v1_stable(tr) or validators.v2_penetration(tr) or validators.v3_visible(scene, tr, lib)
                   or validators.v4_motion(scene, tr, cfg["validation"]["min_motion_m"]) or validators.v7_energy(scene, tr))
        if why is None:
            return scene, tr
        log.append(dict(scene_seed=scene["scene_seed"], attempt=attempt, reason=why))
    raise Rejected(f"scene {split}/{index}: no valid attempt")


def allowed_subtypes(cls, scene, cfg, pol):
    if cls == 3:
        names = [sampler.SCENARIOS[scene["scenario"]][1]]
    else:
        names = [s for s, (c, _, _) in violations.SUBTYPES.items() if c == cls]
    w = cfg["violations"]["subtypes"]
    names = [s for s in names if s is not None and w.get(s, 1.0) > 0]
    if pol["subtypes_heldout"]:
        return [s for s in names if s in pol["heldout_subtypes"]]
    return [s for s in names if s not in pol["heldout_subtypes"]]


def _weighted(rng, d):
    keys = list(d)
    p = np.array([d[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def first_input_diff(A, B, T, S, cfg):
    """First frame whose render inputs (poses at its blur samples, shadow poses, light) differ."""
    n = T * S + 1
    step = np.zeros(n, bool)
    for key in ("pos", "quat", "spos", "squat"):
        step |= (A[key][:, :n] != B[key][:, :n]).any(axis=(0, 2))
    for f in range(T):
        k0, k1 = common.frame_steps(f, S, cfg)
        if step[k0:k1 + 1].any():
            return f
        if any(not np.array_equal(A[k][f], B[k][f]) for k in ("shade_dir", "shadow_dir", "softness", "mode")):
            return f
    return None


def inputs(pos, quat, track):
    return dict(pos=pos, quat=quat, **track)


def clip_meta(scene, cls, role, renderer, v=None, onset=None, offset=None, frac=None):
    T = scene["clip"]["frames"]
    seed = scene["scene_seed"]
    labels = np.zeros(T, int)
    if v is not None:
        labels[onset:offset] = 1
    # test_renderer reuses test_id seeds: tag its ids with the renderer so they never collide
    base = f"s{seed:09d}" + (f"_{renderer}" if scene["split"] == "test_renderer" else "")
    m = dict(clip_id=f"{base}_c{cls}", pair_id=None if cls == 0 else f"{base}_c0_c{cls}",
             scene_seed=seed, generator_version=scene["generator_version"], generator_commit=scene["generator_commit"],
             asset_manifest=scene["asset_manifest"],
             **{"class": cls}, role=role, fps=scene["clip"]["fps"], num_frames=T, resolution=scene["clip"]["resolution"],
             onset_frame=onset, offset_frame=offset, scenario=scene["scenario"], split=scene["split"], renderer=renderer,
             frame_labels=labels.tolist(), frame_classes=(labels * cls).tolist(),
             subtype=None, severity=None, timing=None, involved_objects=[], violation_params=None, visible_fraction=frac,
             objects=[{k: o[k] for k in ("id", "role", "tag", "shape", "scale_cm", "material", "mass_kg", "friction",
                                         "restitution", "texture")} | {"mesh": o["geom"].get("mesh")}
                      for o in scene["objects"]],
             environment=scene["environment"], light=scene["light"], camera=scene["camera"])
    if v is not None:
        m.update(v.meta())
    return m


def try_rng(scene, cls, t):
    """Own stream per violation try, so a stored choice can be replayed exactly with another renderer."""
    return np.random.default_rng(np.random.SeedSequence(
        [scene["scene_seed"], sampler.STREAMS["violation"], scene["attempt"], cls, t]))


def generate_scene(split, index, cfg, lib, renderer, check_rng=None, replay=None):
    """Returns (scene config, [(meta, frames uint8 (T,H,W,3))], rejection log).
    replay: a stored scene config whose violation choices are re-rendered as they are (renderer transfer)."""
    log = []
    scene, tr0 = simulate_c0(split, index, cfg, lib, log)
    if replay is not None and validators.scene_hash(scene) != validators.scene_hash(replay):
        raise Rejected(f"replay of scene {replay['scene_seed']}: regenerated scene differs from the stored config "
                       "(different config, asset manifest or generator version?)")
    T, S = scene["clip"]["frames"], scene["clip"]["substeps"]
    n = T * S + 1
    prep = renderer.prepare(scene, shadow_bounds(scene, tr0, lib, T))
    try:
        track0 = violations.base_track(scene, tr0)
        in0 = inputs(tr0.pos[:, :n], tr0.quat[:, :n], track0)
        frames0 = renderer.render(prep, in0["pos"], in0["quat"], track0, list(range(T)))
        clips = [(clip_meta(scene, 0, "compliant", renderer.name), frames0)]
        pol = sampler.policy(split, cfg)
        compat = sorted(sampler.SCENARIOS[scene["scenario"]][0] & set(cfg["violations"]["classes"]))
        rng = sampler.rng_for(scene["scene_seed"], "violation", scene["attempt"])
        if cfg["dataset"]["pairs"] == "one_random" and compat:
            compat = [compat[rng.integers(len(compat))]]
        scene["violations"] = {}
        vc = cfg["violations"]
        if replay is not None:
            compat = [v["class"] for v in replay.get("violations", {}).values()]
        for cls in compat:
            names = allowed_subtypes(cls, scene, cfg, pol)
            if not names:
                continue
            # weighted order of subtypes; each gets a few tries before the next one, so subtypes that are
            # rarely feasible are not crowded out by the always-feasible ones (balance, V10)
            w = np.array([vc["subtypes"].get(s, 1.0) for s in names], float)
            order = [names[i] for i in rng.choice(len(names), len(names), replace=False, p=w / w.sum())]
            tries = list(enumerate([st for st in order for _ in range(3)][:vc["max_attempts"] * 2]))
            if replay is not None:
                ref = replay["violations"][f"c{cls}"]
                tries = [(ref["try"], ref["subtype"])]
            for attempt, st in tries:
                vrng = try_rng(scene, cls, attempt)
                sev = _weighted(vrng, vc["severity"])
                modes = violations.SUBTYPES[st][2]
                timing = modes[0] if len(modes) == 1 else _weighted(vrng, {m: vc["timing"][m] for m in modes})
                result = _try_violation(st, sev, timing, scene, tr0, in0, frames0, prep, renderer, vrng, cfg, lib,
                                        check_visible=replay is None)
                if isinstance(result, str):
                    log.append(dict(scene_seed=scene["scene_seed"], cls=cls, subtype=st, attempt=attempt, reason=result))
                    continue
                meta, frames, (p, q, tk) = result
                if check_rng is not None and check_rng.uniform() < cfg["validation"]["full_render_check_fraction"]:
                    full = renderer.render(prep, p, q, tk, list(range(meta["onset_frame"])))  # V5 from scratch
                    validators.v5_identical(frames0, full, meta["onset_frame"])
                validators.v5_identical(frames0, frames, meta["onset_frame"])
                validators.v8_labels(meta)
                clips.append((meta, frames))
                scene["violations"][f"c{cls}"] = meta | {"try": attempt}
                break
    finally:
        renderer.release(prep)
    for meta, _ in clips:
        validators.v8_labels(meta)
    return scene, clips, log


def _render_inputs(v, scene, tr0, lib):
    """(pos, quat, track, sim trajectory or None) for a violating clip."""
    T, S = scene["clip"]["frames"], scene["clip"]["substeps"]
    n = T * S + 1
    if v.cls == 1:
        return tr0.pos[:, :n], tr0.quat[:, :n], v.track, None
    if v.cls == 2:
        trv = physics.simulate(scene, T, hooks=v.hooks, mesh_lib=lib)
        return trv.pos[:, :n], trv.quat[:, :n], violations.base_track(scene, trv), trv
    pos, quat = v.traj
    tk = violations.base_track(scene, tr0)
    tk["spos"], tk["squat"] = pos, quat
    return pos[:, :n], quat[:, :n], tk, None


def _try_violation(st, sev, timing, scene, tr0, in0, frames0, prep, renderer, rng, cfg, lib, check_visible=True):
    T, S = scene["clip"]["frames"], scene["clip"]["substeps"]
    v = violations.build(st, sev, timing, scene, tr0, rng, cfg)
    if v is None:
        return "infeasible"
    pos, quat, track, trv = _render_inputs(v, scene, tr0, lib)
    if trv is not None:
        if validators.v1_stable(trv):
            return "unstable violating simulation"
        if hasattr(v, "offset_from_sim"):
            iv = v.offset_from_sim(trv)
            if iv is None:
                return "event did not happen"
            v.offset = iv[1]
    f_start = first_input_diff(in0, inputs(pos, quat, track), T, S, cfg)
    if f_start is None:
        return "no change in render inputs"
    frames = frames0.copy()
    frames[f_start:] = renderer.render(prep, pos, quat, track, list(range(f_start, T)))
    changed = np.nonzero((frames != frames0).any(axis=(1, 2, 3)))[0]
    if not len(changed):
        return "V6 no pixel change"
    onset = int(changed[0])
    if v.offset is None or v.offset <= onset:
        return f"onset {onset} not before offset {v.offset}"
    why, frac = validators.v6_visible(frames0, frames, onset, v.offset, cfg["validation"]["visibility_threshold"][v.severity])
    if why and check_visible:
        return why
    meta = clip_meta(scene, v.cls, "violating", renderer.name, v, onset, int(v.offset), frac)
    return meta, frames, (pos, quat, track)
