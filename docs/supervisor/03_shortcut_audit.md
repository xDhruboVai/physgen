# Shortcut audit of the pilot dataset

A synthetic dataset is only useful if a model has to look at the physics to get the labels right. If
the generator leaves fingerprints, for example violating clips that are blurrier, have more motion,
use different objects or always start their violation at the same time, a model can score well
without learning anything about physics. This audit checks for that, in two parts: first whether the
data is what it claims to be, then whether simple cues can predict the labels.

## 1. Is the data what it claims to be?

I ran the `validate` command over the finished pilot (all 10,290 clips, 7,360 pairs). It found no
errors and no warnings. In practice that means:

- every clip has 64 frames and its per-frame labels agree with its onset and offset;
- in every pair, the stored frames before the onset are byte-identical to the compliant clip, so the
  onset label really is the first moment the two clips differ;
- there are no near-duplicate scenes, and no stray files from interrupted runs.

Checks also ran while generating. Scenes were redrawn when the objects would have been too small to
see (303 drafts) or when objects interpenetrated by more than 5 mm (14 drafts). Violations were redrawn
when they were too faint to see (2,230 drafts) or when the event they needed, such as a bounce or a
collision, did not happen (137 drafts). The visibility threshold matters later on: it means
every labelled violation changes a noticeable part of the image.

## 2. Can cheap cues predict the label?

For each probe I gave a plain logistic regression a deliberately limited view of the clip and asked
it to predict the label. If any of these simple models works, there is a shortcut. All probes use
5-fold cross-validation grouped by scene, so the two clips of a pair are never split between training
and testing. They ran on 4,000 clips (all clips of a random subset of scenes) and report balanced
accuracy, where chance is 0.50 for "compliant or violating" and 0.25 for the four classes.

| Probe | What the model sees | Compliant vs violating | Four classes |
|---|---|---|---|
| P1 | one frame from the middle of the clip | 0.50 | 0.27 |
| P2 | the first 8 frames (before any onset) | 0.50 | 0.27 |
| P3 | motion energy and optical-flow statistics only | 0.50 | 0.35 |
| P4 | metadata: scenario, shapes, materials, floor, environment map | 0.50 | 0.27 |
| P6 | image sharpness and JPEG size per frame | 0.50 | 0.30 |
| P5 | onset frame only (violating clips, three classes, chance 0.33) | | 0.34 |
| P5b | interval length only (for information; c3 is short by design) | | 0.32 |

## 3. What the numbers say

**The main question is clean.** No probe can tell a violating clip from a compliant one better than a
coin flip. This is mostly the paired design at work: both clips of a pair share the scene, the objects,
the look and the camera, and they are identical until the violation starts.

**Onset position is not a cue.** This was not true at first. In an early 150-scene test run, the onset
frame alone predicted the class with 0.42 balanced accuracy, because the causality violations (c3) all
happened early: their cause-and-effect contacts were placed in the first half of the clip. I spread
the contact times so that c3 onsets cover the same window as the other classes, and the probe dropped
to 0.34, which is chance.

**Appearance does not give the class away.** A single frame (P1) and the metadata (P4) stay close to
chance on the four-class question. Shadow errors (c1) might in principle be visible in a single
frame; a linear model on small thumbnails does not pick that up.

**Motion statistics are the one weak spot.** The motion probe (P3) reaches 0.35 on the four-class
question and the image-statistics probe (P6) 0.30. My first guess was that these probes simply
recognise the scenario, since c3 only exists in billiards and ramp scenes. That turned out not to be
the whole story. When I trained the same probes inside a single scenario, they still did better than
chance:

| Scenario | Classes | P3 motion | P6 image statistics | P1 single frame | Chance |
|---|---|---|---|---|---|
| billiards | 4 | 0.31 | 0.29 | 0.27 | 0.25 |
| ramp slide | 4 | 0.38 | 0.28 | 0.26 | 0.25 |
| free fall | 3 | 0.43 | 0.52 | 0.36 | 0.33 |

Looking at free fall in detail: before the onset, the motion energy of all classes is the same, as
it should be. After the onset, a violating clip changes the image more than its own compliant twin
in 85% of shadow pairs (the shadow moves) and 69% of mechanics pairs (objects move differently). So
the probes are not finding a fingerprint of the generator; they are finding the violation itself,
seen through a very crude statistic. That is still a problem. A model could learn "a lot of change
late in the clip means mechanics" without learning when a motion is impossible. The paired design
keeps the compliant-vs-violating question safe, because this extra change is small next to the
differences between scenes, but the class question is partly exposed.

## 4. Problems the checks caught during development

Besides the onset timing above, two issues were found and fixed before the final data was generated:

- While comparing difference images, I noticed that in mechanics and causality clips the whole floor
  changed by about one grey level. My ambient-occlusion approximation let every object darken every
  pixel a tiny amount, so moving one object changed the entire image. That faint global change could
  have become a cue. I limited occlusion to each object's neighbourhood; differences are now local.
- My first comparison between the two renderers measured the Blender clips against the whole of
  test_id. Measured fairly, on the same 30 scenes in both renderers, there is no clear gap. The
  results document reports the fair comparison.

## 5. Limits of this audit

The probes are deliberately simple: linear models on small summaries. A deep network can find cues
that they miss, so "near chance" here is necessary but not sufficient. The probes also ran on a
4,000-clip subset, and every number above comes from one cross-validation run.

## 6. What I will change for the next dataset (M2)

1. Add compliant clips with as much late motion as violating ones (the design document's "hard c0
   cases": fast, bouncy or heavy objects), and give more weight to violations that remove motion
   (mid-air stop, sudden stop by friction), so that "more change" stops pointing at a violation.
2. Rebalance the subtypes. The rare ones (mid-air stop, bounce, swapped masses) are also the ones that
   change motion in unusual ways.
3. Add stronger probes: the baseline model itself trained on single frames, on pre-onset frames only
   and on frame-difference energy only. Any of these doing well would point to a shortcut a linear
   probe cannot see.
4. Report probe results per scenario as standard, and add a paired probe that compares each violating
   clip only with its own compliant twin.
