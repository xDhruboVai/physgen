# physgen: physics-violation video dataset generator

Implements *Physics-Violation Dataset Generator: Design and Data Documentation* (draft v0.1).
Every scene yields a physics-compliant clip (c0) and up to three violating clips from the same
seed (c1 shadow/lighting, c2 rigid-body mechanics, c3 causality), with exact frame-level onset
and offset labels.

All tools, assets and Python packages live on E: (`.venv`, `third_party/`, `assets/`).

## Quick start

```bash
.venv/Scripts/python -m physgen fetch-assets            # Poly Haven CC0 textures + HDRIs, procedural patterns
.venv/Scripts/python -m physgen fetch-assets --ycb 35   # optional: YCB meshes (D7 = B)
.venv/Scripts/python -m physgen preview --split train --index 0 --scenario billiards --out previews
.venv/Scripts/python -m physgen generate --config configs/pilot.yaml --split train --workers 6 --out dataset_root
.venv/Scripts/python -m physgen generate --config configs/pilot.yaml --split val --num-scenes 200 --out dataset_root
.venv/Scripts/python -m physgen validate --root dataset_root
.venv/Scripts/python -m physgen audit --root dataset_root
.venv/Scripts/python -m physgen export-features --root dataset_root --model MCG-NJU/videomae-base
.venv/Scripts/python tests/test_pipeline.py
```

`generate` resumes: rerunning the same command skips scenes whose shard was committed.
Disjoint `--start` ranges can run on several machines and be merged into one `dataset_root`.

## Options (every decision in the doc is a config switch)

Defaults are in `physgen/default.yaml`; a config file overrides only what it names.
Presets: `configs/pilot.yaml` (M1), `configs/full.yaml` (M2-M4), `configs/blender_eval.yaml`
(renderer transfer), `configs/vjepa2_256.yaml` (another backbone's input size).

| Doc | Choice implemented (default) | Alternatives | Config key |
|---|---|---|---|
| D1 renderer | GPU rasterizer (moderngl, own shadow pass) | Blender Cycles / EEVEE | `renderer.backend: raster \| blender` |
| D2 physics | PyBullet DIRECT, 240 Hz, 4 internal substeps | new engines plug into `physics.BACKENDS` | `physics.*` |
| D3 pairs | up to three pairs per scene | one random violation per scene | `dataset.pairs: compatible_all \| one_random` |
| D4 clip | 64 frames, 24 fps, 224 px | any frames/fps/resolution (e.g. 16 frames, 256 px) | `dataset.clip` |
| D5 timing | mixed persistent/transient per subtype | any mix | `violations.timing` |
| D6 storage | JPEG q95 frames in WebDataset tars | PNG, NPY, MP4 (+ backbone features) | `output.format`, `export-features` |
| D7 assets | procedural primitives | + YCB meshes (`mesh_fraction` 0.4) | `assets.meshes`, `assets.mesh_fraction` |
| D8 severity | subtle/clear/strong 0.3/0.4/0.3 | any mix; ranges in `violations.SEVERITY` | `violations.severity` |
| D9 camera | static (pilot) | slow orbit/pan for a fraction of scenes | `camera.moving_fraction` |
| D10 scale | `num_scenes` | | `dataset.num_scenes`, `--num-scenes` |
| D11 augmentation | motion blur + exposure baked | blur off with `motion_blur_samples: 1` | `renderer.motion_blur_samples`, `renderer.exposure_ev` |

Changing the backbone needs no regeneration when its input fits the stored frames: windows and
stride are chosen at training or feature-export time, and `export-features --model <any HF id>`
writes features to their own folder. Only a different resolution or frame rate needs `dataset.clip`.

Other switches: scenario weights (`scenarios`), subtype weights (`violations.subtypes`, 0 disables),
held-out shapes/scenarios/subtypes for the OOD splits (`splits`), onset window (`violations.onset_range`),
visibility thresholds (`validation.visibility_threshold`).

## Splits (doc 5.6)

`--split` is one of `train, val, test_id, test_ood_objects, test_ood_env, test_ood_scenario,
test_ood_subtype, test_renderer`. Seeds are disjoint per split; held-out assets are 15% of
textures, HDRIs and meshes (stable hash of the asset id). `test_renderer` uses the `test_id` seeds
and, when `configs/test_id/` exists in the same root, replays exactly the stored violations with
the configured renderer (it refuses if the regenerated scene differs from the stored one).

## Output (doc 5.5)

```
dataset_root/
  shards/<split>/shard-<run>-<k>.tar   <clip_id>.000.jpg ... <clip_id>.063.jpg, <clip_id>.json
  manifest/clips.parquet               one row per clip (all metadata; nested fields as JSON)
  manifest/pairs.parquet               one row per (c0, violating) pair
  configs/<split>/s<seed>.json         full scene config incl. chosen violations
  stats/                               balance tables, probes.csv
  features/<model>-w<W>-s<S>/          optional backbone features + index.parquet
  logs/                                rejected attempts and errors per run
```

Clip metadata follows doc 5.4 (`clip_id, pair_id, class, subtype, severity, timing, onset_frame,
offset_frame, frame_labels, frame_classes, involved_objects, objects, environment, light, camera,
violation_params, ...`). Intervals are half-open `[onset_frame, offset_frame)`.
Ids are `s<seed>_c<class>` and `s<seed>_c0_c<class>`; `test_renderer` ids carry the renderer
(`s<seed>_blender_c1`) because they share seeds with `test_id`.
`onset_frame` is the first frame whose pixels differ from c0; frames before it are byte-identical
to c0 (copied, and checked by V5 in `validate`).

## Violations (doc 6.3)

- c1 (`S1.1`-`S1.6`) edits only the shadow light (direction, elevation, softness) or the
  shadow-caster poses; the visible trajectories are the c0 ones.
- c2 (`S2.1`-`S2.8`) edits the simulation at the injection step and lets it continue.
  `S2.2` is an instantaneous velocity kick (a force spread over frames is absorbed by static friction).
  `S2.5` uses the velocity-boost fallback for restitution above 1.
- c3 (`S3.1`-`S3.4`) time-advances the effect trajectory (and everything it touches later) by
  delta = 4-14 frames; the interval is `[t_c - delta, t_c]`.

Scenario compatibility follows doc 6.4. Infeasible choices (e.g. no airborne object for S2.4)
are retried with other parameters or subtypes; each attempt is logged in `logs/`.

## Validation and audit (doc 9)

At generation: V1 stability, V2 penetration (5 mm), V3 visibility and readable size (14 px median),
V4 minimum motion, V5 pre-onset identity, V6 visibility per severity, V7 energy, V8 labels.
`validate` re-checks V5/V8 on the stored files, near-duplicates (V9), orphan shards and prints
class/subtype counts (V10). `audit` writes balance tables and runs probes P1-P6 (logistic
regression, grouped by scene) and P5b (interval duration, informational).

## Measured performance (this PC: Ryzen 5 5600, GTX 1050 Ti 4 GB)

Measured with `tests/benchmark.py` and the generation logs:

| Stage | Metric | Measured |
|---|---|---|
| Simulation | c0 simulations per second, one core (64+16 frames, 960 Hz solver) | 5.3 |
| Rendering (raster) | frames per second, 224 px, 4 motion-blur samples | 66 |
| Rendering (Blender Cycles, 32 spp, OptiX, per-pass OIDN) | seconds per frame (+~1 s per call) | 1.5 |
| Writing (JPEG q95) | clips per second, one core | 44 |
| End to end (raster, 6 workers) | pairs per hour | ~6,300 (150 scenes in 216 s) |
| End to end (Blender, 1 worker) | seconds per scene (c0 + 3 violations) | ~245 |
| Storage (JPEG q95) | MB per clip | 1.44 |

So the M1 pilot (~5k pairs) takes about an hour on this PC, and ~26 GB per 5k scenes.

## Baseline (doc section 10)

Frozen VideoMAE V2 encoder + trainable temporal head (`baseline/`). Features are exported once per
backbone; the head sees one feature per 2 frames (tubelet) over the whole clip, assembled from the
16-frame windows, and predicts {normal, c1, c2, c3} per time step.

The pilot export uses stride 16 (non-overlapping windows) instead of the doc's stride 8: on the
GTX 1050 Ti the encoder runs at ~3.8 windows/s, and the per-2-frame timeline keeps the same temporal
resolution either way. Windows that end before a violating clip's onset are copied from its c0
clip (identical frames, identical features). Use `--stride 8` on a faster GPU.

```bash
.venv/Scripts/python -m physgen export-features --root datasets/pilot --stride 16   # VideoMAE V2-Base
.venv/Scripts/python -m baseline train --root datasets/pilot --out results/head [--pairwise]
.venv/Scripts/python -m baseline evaluate --root datasets/pilot --run results/head
```

`evaluate` reports, per split: clip AUROC and accuracy, pairwise ranking accuracy (violating clip
scored above its c0 partner), 4-class macro F1, onset/offset error in frames, mean temporal IoU and
segment F1 at IoU 0.3/0.5/0.7, plus breakdowns by class, subtype, severity, scenario, timing and
camera mode. Thresholds are tuned on val. VideoMAE V2 weights (CC BY-NC 4.0) are loaded into the
built-in transformers VideoMAE model without running repository code (`tests/test_videomaev2.py`
checks the mapping against an independent reference). Fine-tuning the encoder does not fit on a 4 GB GPU.

The real / AI-generated test set needs data collected by hand; see `docs/real_test_protocol.md`.

## Pilot results

The M1 pilot dataset (`docs/pilot/dataset_card.md`) and baseline results (`results/README.md`):
frozen VideoMAE V2 + temporal head reaches test_id AUROC 0.70-0.75 and segment F1@0.5 up to 0.38;
difficulty follows the severity levels, and OOD splits drop as expected.

## Known limits and open items

- Physics backends: only PyBullet is implemented; the operations c2 needs are isolated in
  `physics.PyBulletBackend` so MuJoCo can be added as another class.
- S2.6 (mass inconsistency) is only detectable when the masses were observable earlier in the clip;
  results per subtype will show this. c3 strikers still react at `t_c` (doc 13, open item).
- Some subtypes are rarely feasible in some scenarios (S2.4, S2.5, S2.6); check `validate` counts
  and raise their weight in `violations.subtypes` if a quota matters.
- `audit` P3 (4-class) and P5b sit above chance because c3 exists only in some scenarios and its
  interval is short by definition; P5 (onset only) should stay near chance.
- The Blender backend approximates S1.6 by changing the light size per frame; with EEVEE, c1
  direction edits use a negative light and are approximate.
- pybullet has no Windows pip wheels; it was copied into `.venv` from the conda-forge build.
