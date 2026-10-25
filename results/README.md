# Pilot results (M1)

Dataset: `docs/pilot/dataset_card.md`. 1,900 train scenes (4,934 pairs), 200 val, 200 test_id,
150 per OOD split, 30 Blender scenes (`test_renderer`); free fall, ramp slide and billiards;
static camera; procedural primitives. Generated at commit 59bfd1e (Blender subset at 666f303,
same generator code). Validation: 10,290 clips, 0 errors (pre-onset frames byte-identical).

Model: frozen VideoMAE V2-Base (16-frame windows, stride 16, one feature per 2 frames) +
2-layer transformer head predicting {normal, c1, c2, c3} per time step. Thresholds tuned on val.
Full tables: `pilot_head/metrics.md` (cross-entropy only) and `pilot_head_pairwise/metrics.md`
(+ ranking loss: violating clip above its c0 partner).

## Main numbers (test_id)

| Metric | Independent | + pairwise loss |
|---|---|---|
| AUROC (c0 vs violating) | 0.698 | **0.751** |
| Pairwise ranking accuracy | 0.717 | **0.805** |
| 4-class macro F1 | 0.526 | **0.577** |
| Mean temporal IoU | **0.405** | 0.359 |
| Onset error (frames, localised clips) | **7.2** | 8.3 |
| Segment F1 @ IoU 0.3 / 0.5 / 0.7 | **0.475 / 0.379 / 0.168** | 0.468 / 0.291 / 0.142 |

## Generalisation (AUROC / segment F1@0.5, independent head)

| Split | AUROC | Seg F1@0.5 |
|---|---|---|
| test_id | 0.698 | 0.379 |
| test_ood_env (unseen textures, HDRIs) | 0.709 | 0.353 |
| test_ood_scenario (projectile into bin) | 0.687 | 0.360 |
| test_renderer (Blender Cycles) | 0.671 | 0.333 |
| test_ood_objects (unseen shape family) | 0.640 | 0.241 |
| test_ood_subtype (S1.5, S2.7 zero-shot) | 0.623 | 0.171 |

## Observations

- The task is learnable but far from solved with a frozen encoder: train AUROC 0.92-0.98 vs val
  0.72-0.74, and val stops improving around epoch 27-46. Fine-tuning the encoder (doc 10 ablation)
  does not fit on a 4 GB GPU.
- Difficulty follows the severity levels (test_id ranking accuracy, independent head): strong 0.83,
  clear 0.73, subtle 0.59.
- By class: causality (c3) and shadows (c1) are easier than mechanics (c2). Hardest subtypes: S2.2
  instantaneous kick (3-frame interval) and S2.6 mass inconsistency (class accuracy 0, as the design
  doc anticipated: masses are rarely observable before the collision).
- The pairwise loss helps detection and classification (+5 AUROC, +9 ranking) but blurs temporal
  boundaries (-9 points segment F1@0.5): a trade-off worth reporting in the ablation.
- OOD drops are largest for unseen objects and zero-shot subtypes; the renderer gap (raster to
  Cycles) costs about 3 AUROC points.
- Shortcut probes stay at chance for c0-vs-violating (`docs/pilot/probes.csv`); the 4-class motion
  probe (0.35 vs 0.25) reflects that c3 exists only in billiards and ramp scenes.

## Reproduce

```bash
.venv/Scripts/python -m physgen generate --config configs/pilot.yaml --split train --num-scenes 1900 --out datasets/pilot
# ... val/test_id 200, OOD splits 150, test_renderer 30 with --renderer blender (see README)
.venv/Scripts/python -m physgen export-features --root datasets/pilot --stride 16 --batch 12
.venv/Scripts/python -m baseline train --root datasets/pilot --out results/pilot_head [--pairwise]
.venv/Scripts/python -m baseline evaluate --root datasets/pilot --run results/pilot_head
```
