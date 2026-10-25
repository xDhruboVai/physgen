# Physics-violation video dataset

Synthetic minimal pairs for detecting and temporally localising physics violations. Every scene
yields a physics-compliant clip (class 0) and up to three violating clips from the same seed:
1 shadow/lighting, 2 rigid-body mechanics, 3 causality (effect before cause).
Frames before the labelled onset are identical to the compliant clip.

## Provenance

- Generator: physgen 0.1.0, commit(s) 59bfd1e, 666f303
- Asset manifest hash(es): f54d6907c8ea
- Renderer(s): blender, raster
- Clip format: 64 frames at 24 fps, 224 px, jpeg frames in WebDataset tar shards
- Every clip can be regenerated from its scene config in `configs/<split>/` with the recorded commit.

## Contents

| split             |    0 |    1 |    2 |    3 |
|:------------------|-----:|-----:|-----:|-----:|
| test_id           |  200 |  200 |  192 |  141 |
| test_ood_env      |  150 |  147 |  141 |   99 |
| test_ood_objects  |  150 |  147 |  139 |  103 |
| test_ood_scenario |  150 |  150 |  140 |    0 |
| test_ood_subtype  |  150 |  113 |  113 |    0 |
| test_renderer     |   30 |   30 |   30 |   24 |
| train             | 1900 | 1877 | 1817 | 1240 |
| val               |  200 |  196 |  193 |  128 |

Pairs per subtype and severity:

| subtype   |   clear |   strong |   subtle |
|:----------|--------:|---------:|---------:|
| S1.1      |     294 |      204 |      229 |
| S1.2      |     296 |      138 |      241 |
| S1.3      |     196 |       92 |      185 |
| S1.4      |     276 |      188 |      205 |
| S1.5      |      47 |       27 |       39 |
| S1.6      |      87 |       59 |       57 |
| S2.1      |     139 |       34 |      200 |
| S2.2      |     300 |      209 |      242 |
| S2.3      |     219 |      345 |      173 |
| S2.4      |      63 |       31 |       55 |
| S2.5      |      79 |       60 |       59 |
| S2.6      |      66 |       59 |       56 |
| S2.7      |      49 |       33 |       31 |
| S2.8      |     106 |       71 |       86 |
| S3.1      |     366 |      241 |      244 |
| S3.4      |     378 |      192 |      314 |

Pairs per scenario and class:

| scenario       |   1 |   2 |   3 |
|:---------------|----:|----:|----:|
| billiards      | 922 | 894 | 851 |
| free_fall      | 812 | 817 |   0 |
| projectile_bin | 150 | 140 |   0 |
| ramp_slide     | 976 | 914 | 884 |

## Splits

Split by scene seed; held-out assets (~15% of textures, HDRIs and meshes), shape families, scenarios
and subtypes appear only in their OOD split. `test_renderer` re-renders `test_id` scenes with another
renderer (ids carry the renderer name).

- Held-out shape families: ['capsule']
- Held-out scenarios: ['projectile_bin']
- Held-out subtypes: ['S1.5', 'S2.7', 'S3.3']
- Held-out asset ids: `splits/heldout.json` (33 assets)

## Labels and files

- `shards/<split>/*.tar`: `<clip_id>.<frame>.jpg` plus `<clip_id>.json` (all metadata).
- `manifest/clips.parquet` (one row per clip), `manifest/pairs.parquet` (one row per pair).
- `onset_frame`, `offset_frame`: half-open interval `[onset, offset)`; `frame_labels` (0/1) and
  `frame_classes` (0-3) per frame; `subtype`, `severity` (subtle/clear/strong), `timing`
  (persistent/transient/event), `involved_objects`, `violation_params`.

## Shortcut probes (balanced accuracy; should stay near chance)

| probe   | input                                                        |   binary_bal_acc |   four_class_bal_acc |   chance_binary |   chance_classes |
|:--------|:-------------------------------------------------------------|-----------------:|---------------------:|----------------:|-----------------:|
| P1      | single frame                                                 |         0.500626 |             0.274515 |             0.5 |         0.25     |
| P2      | first 8 frames (pre-onset)                                   |         0.5      |             0.266596 |             0.5 |         0.25     |
| P3      | motion energy / optical flow                                 |         0.500714 |             0.347891 |             0.5 |         0.25     |
| P4      | metadata (scenario, shapes, materials, floor, hdri)          |         0.501062 |             0.268637 |             0.5 |         0.25     |
| P6      | high-frequency / compression stats                           |         0.5      |             0.299593 |             0.5 |         0.25     |
| P5      | onset position only (violating clips, 3-class)               |       nan        |             0.342137 |             0.5 |         0.333333 |
| P5b     | interval duration (informational: c3 is short by definition) |       nan        |             0.317314 |             0.5 |         0.333333 |

## Licences

- Textures and HDRIs: Poly Haven, CC0. Procedural patterns: generated, CC0.
- YCB object meshes (if used): Calli et al., *The YCB Object and Model Set*, CC BY 4.0 - attribution required.
- Frozen-feature exports from VideoMAE V2 weights inherit their CC BY-NC 4.0 licence.

## Known limitations

- Mass inconsistency (S2.6) is only detectable when masses were observable earlier in the clip.
- In c3 the cause object still reacts at the true contact time (design caveat, doc section 6.3).
- c3 exists only in scenarios with a cause-effect interaction, so scenario and class correlate.
