"""Self-checks for the generator. Run: python tests/test_pipeline.py  (or pytest)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from physgen import config, pipeline, sampler, shapes, validators, violations, writer  # noqa: E402
from physgen.assets import AssetLibrary  # noqa: E402
from physgen.render import get_renderer  # noqa: E402

CFG = config.load(None, {"dataset": {"clip": {"frames": 48}}})
LIB = AssetLibrary("assets")
_REN = {}


def renderer():
    if "r" not in _REN:
        _REN["r"] = get_renderer(CFG, LIB)
    return _REN["r"]


def only(scenario):
    return config.merge(CFG, {"scenarios": {k: float(k == scenario) for k in sampler.SCENARIOS}})


def test_mass_props_box():
    h = np.array([0.1, 0.05, 0.02])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h
    vol, com, I = shapes.mass_props(shapes.hull_tris(corners))
    assert np.isclose(vol, 8 * h.prod()) and np.allclose(com, 0, atol=1e-12)
    a, b, c = 2 * h
    assert np.allclose(np.diag(I) / vol, [(b * b + c * c) / 12, (a * a + c * c) / 12, (a * a + b * b) / 12])


def test_scene_is_pure_function_of_seed():
    cfg = only("billiards")
    a, ta = pipeline.simulate_c0("train", 7, cfg, LIB, [])
    b, tb = pipeline.simulate_c0("train", 7, cfg, LIB, [])
    assert writer.to_json(a) == writer.to_json(b)
    assert np.array_equal(ta.pos, tb.pos) and np.array_equal(ta.quat, tb.quat)


def test_pre_onset_identity_full_render_and_labels():
    """V5: rendering a violating clip from frame 0 reproduces c0 exactly before the onset (the label is exact)."""
    for scenario in ("ramp_slide", "dominoes"):
        cfg = only(scenario)
        scene, clips, _ = pipeline.generate_scene("train", 4, cfg, LIB, renderer())
        assert len(clips) >= 2, f"{scenario}: no violating clip"
        frames0 = clips[0][1]
        scene2, tr0 = pipeline.simulate_c0("train", 4, cfg, LIB, [])
        T = scene2["clip"]["frames"]
        prep = renderer().prepare(scene2, pipeline.shadow_bounds(scene2, tr0, LIB, T))
        try:
            again = renderer().render(prep, tr0.pos, tr0.quat, violations.base_track(scene2, tr0), list(range(T)))
            assert np.array_equal(again, frames0), "c0 render is not deterministic"
            for meta, frames in clips[1:]:
                validators.v8_labels(meta)
                on, off = meta["onset_frame"], meta["offset_frame"]
                assert np.array_equal(frames[:on], frames0[:on])
                assert not np.array_equal(frames[on], frames0[on]), "onset frame must differ"
                assert meta["frame_labels"][on] == 1 and (off == T or meta["frame_labels"][off] == 0)
        finally:
            renderer().release(prep)


def test_violation_rng_replay():
    """test_renderer replays a stored violation choice exactly (same params, same labels)."""
    cfg = only("billiards")
    scene, clips, _ = pipeline.generate_scene("train", 3, cfg, LIB, renderer())
    stored = writer.to_json(scene)
    import json
    scene_r, clips_r, _ = pipeline.generate_scene("train", 3, cfg, LIB, renderer(), replay=json.loads(stored))
    for (m1, f1), (m2, f2) in zip(clips, clips_r):
        assert m1["violation_params"] == m2["violation_params"] and m1["onset_frame"] == m2["onset_frame"]
        assert np.array_equal(f1, f2)


def test_encode_decode_roundtrip():
    frames = (np.random.default_rng(0).uniform(0, 255, (4, 32, 32, 3))).astype(np.uint8)
    for fmt in ("png", "npy"):
        enc = writer.encode(frames, config.merge(CFG, {"output": {"format": fmt}}))
        assert np.array_equal(writer.decode(enc), frames)
    enc = writer.encode(frames, config.merge(CFG, {"output": {"format": "jpeg"}}))
    assert writer.decode(enc).shape == frames.shape


def test_labels_validator_rejects_bad_offset():
    meta = dict(num_frames=10, frame_labels=[0] * 10, frame_classes=[0] * 10, onset_frame=3, offset_frame=11, **{"class": 2})
    meta["frame_labels"][3:] = [1] * 7
    try:
        validators.v8_labels(meta)
    except AssertionError:
        return
    raise AssertionError("V8 accepted an offset beyond the clip")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
