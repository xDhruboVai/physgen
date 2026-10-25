# Pilot results (milestone M1)

## Setup

The baseline is the one planned in the design document: a frozen video encoder with a small trainable
temporal head. The encoder is VideoMAE V2-Base. Each clip is cut into 16-frame windows, and the
encoder's tokens are averaged over space only, which leaves one feature vector for every 2 frames of
the clip. A two-layer transformer reads this 32-step timeline and predicts, for every step, whether it
is normal or which violation class it shows. A clip's violation score is its most confident step, and
the predicted interval is the strongest run of steps above a threshold. Both thresholds were tuned on
the validation split only.

I trained two versions on the 4,934 training pairs:

- **independent**: per-step cross-entropy only;
- **pairwise**: the same plus a ranking term that pushes every violating clip's score above the score of
  its own c0 clip (the "pairwise vs independent" ablation from the design document).

Each version is a single training run (seed 0), so differences of a few points between them should
be read with that in mind. Feature windows were non-overlapping (stride 16) because the GTX 1050 Ti
manages only about 4 windows per second; the per-2-frame timeline is the same either way.

## Main table

Higher is better except for onset error. AUROC and accuracy are for c0 vs violating; ranking is the
share of pairs where the violating clip scores above its c0 partner; macro F1 is over the four
classes; tIoU and segment F1 measure how well the predicted interval matches the labelled one;
onset error is in frames, over clips where an interval was found.

| Split | Head | AUROC | Ranking | Macro F1 | tIoU | Onset err. | Seg. F1 @0.3 | @0.5 | @0.7 |
|---|---|---|---|---|---|---|---|---|---|
| val | independent | 0.722 | 0.727 | 0.540 | 0.405 | 6.9 | 0.478 | 0.346 | 0.177 |
| | pairwise | 0.735 | 0.820 | 0.569 | 0.333 | 8.5 | 0.418 | 0.272 | 0.144 |
| test_id | independent | 0.698 | 0.717 | 0.526 | 0.405 | 7.2 | 0.475 | 0.379 | 0.168 |
| | pairwise | **0.751** | **0.805** | **0.577** | 0.359 | 8.3 | 0.468 | 0.291 | 0.142 |
| test_ood_env | independent | 0.709 | 0.726 | 0.510 | 0.408 | 6.9 | 0.481 | 0.353 | 0.178 |
| | pairwise | 0.700 | 0.734 | 0.520 | 0.342 | 8.6 | 0.420 | 0.269 | 0.126 |
| test_ood_scenario | independent | 0.687 | 0.666 | 0.434 | 0.453 | 8.5 | 0.494 | 0.360 | 0.195 |
| | pairwise | 0.651 | 0.710 | 0.413 | 0.411 | 9.5 | 0.421 | 0.253 | 0.152 |
| test_ood_objects | independent | 0.640 | 0.632 | 0.419 | 0.385 | 7.3 | 0.351 | 0.241 | 0.121 |
| | pairwise | 0.650 | 0.692 | 0.464 | 0.317 | 8.2 | 0.365 | 0.240 | 0.104 |
| test_ood_subtype | independent | 0.623 | 0.593 | 0.262 | 0.368 | 9.2 | 0.242 | 0.171 | 0.106 |
| | pairwise | 0.670 | 0.712 | 0.309 | 0.379 | 10.2 | 0.279 | 0.190 | 0.110 |
| test_renderer (Blender) | independent | 0.671 | 0.667 | 0.492 | 0.430 | 6.9 | 0.470 | 0.333 | 0.167 |
| | pairwise | 0.683 | 0.714 | 0.478 | 0.411 | 8.5 | 0.391 | 0.299 | 0.115 |
| train | independent | 0.916 | 0.932 | 0.789 | 0.631 | 4.5 | 0.810 | 0.735 | 0.468 |
| | pairwise | 0.983 | 0.999 | 0.854 | 0.516 | 6.9 | 0.797 | 0.562 | 0.284 |

test_ood_scenario and test_ood_subtype contain no c3 clips, which pulls their 4-class macro F1 down.
The same numbers are in `02_pilot_results.csv`; breakdowns for every split are in the repository under `results/`.

## Where the model succeeds and fails (test_id, independent head)

| Severity | Pairs | Ranking | Detected | Class correct | tIoU |
|---|---|---|---|---|---|
| strong | 153 | 0.83 | 0.61 | 0.57 | 0.48 |
| clear | 224 | 0.73 | 0.54 | 0.43 | 0.41 |
| subtle | 156 | 0.59 | 0.31 | 0.25 | 0.32 |

| Class | Pairs | Ranking | Detected | Class correct | tIoU |
|---|---|---|---|---|---|
| c1 shadow / lighting | 200 | 0.74 | 0.52 | 0.49 | 0.49 |
| c2 mechanics | 192 | 0.65 | 0.39 | 0.27 | 0.27 |
| c3 causality | 141 | 0.78 | 0.60 | 0.52 | 0.47 |

By subtype, the easiest cases are a shadow pointing the wrong way (S1.1, ranking 0.81), a wrong shadow
length (S1.4, 0.84) and an object sliding before it is released (S3.4, 0.82). The hardest are the
push with no contact (S2.2, ranking 0.49, which is chance) and swapped masses (S2.6, never classified
correctly). Both make sense: S2.2 lasts only three frames, and S2.6 can only be noticed if the
masses were visible from an earlier collision.

## Renderer transfer, compared fairly

The Blender split contains 30 test_id scenes rendered a second time. Comparing the model on exactly
those 30 scenes in both renderers:

| Head | Raster AUROC | Blender AUROC | Raster tIoU | Blender tIoU |
|---|---|---|---|---|
| independent | 0.634 | 0.671 | 0.377 | 0.430 |
| pairwise | 0.700 | 0.683 | 0.358 | 0.411 |

So far there is no visible drop when the renderer changes, but this is only 84 pairs and the
uncertainty on these numbers is around plus or minus 0.1. The M2 plan has 200 Blender scenes.

## What I take from this

1. The task is learnable from the data but far from solved with a frozen encoder. The model fits the
   training set well (train AUROC 0.92 to 0.98) while validation stalls around 0.72 to 0.74 after about
   30 epochs. That gap could come from the frozen features or from too little data; fine-tuning the
   encoder (which did not fit on the 4 GB card) and the larger M2 dataset will tell the two apart.
2. The severity levels behave as intended: performance drops steadily from strong to subtle. That is
   useful for the thesis because it gives a difficulty curve rather than one number.
3. The pairwise loss is a real trade-off. It helps detection (+5 AUROC and +9 points of ranking on
   test_id) but makes the predicted intervals less precise (-9 points of segment F1 at IoU 0.5).
4. Out-of-distribution splits behave as expected. New textures and environments barely matter, a new
   scenario costs a little, and unseen shapes and unseen violation types cost the most. The
   zero-shot subtype split is the hardest (AUROC 0.62, ranking 0.59 for the independent head).
