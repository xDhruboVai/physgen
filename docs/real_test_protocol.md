# Annotation protocol for the real / AI-generated test set (`test_real`)

Draft for doc section 10 ("The real set needs a written annotation protocol"). Adjust before labelling starts.

## 1. What goes in the set

- Clips of 2-6 s with rigid objects (no fluids, cloth, people or animals; same scope as the synthetic data).
- Sources: (a) real footage you record or that has a licence allowing research use, (b) AI-generated
  videos (record the model, version, prompt and date for each clip).
- Aim for roughly equal numbers of clips with and without a violation, and cover all three classes.
- Store each clip as MP4 plus one row in `labels.csv` (below). Keep the original resolution;
  resizing to the model input happens at evaluation time.

## 2. Who labels

- Two annotators label every clip independently, without seeing each other's labels or any model output.
- A third person resolves disagreements (class differs, or onset/offset differ by more than 6 frames at 24 fps).
- Report agreement: Cohen's kappa for class, mean absolute onset difference in frames.

## 3. What counts as a violation

Label only what a careful viewer can see at normal playback speed (rewatching is allowed).

| Class | Counts | Does not count |
|---|---|---|
| 1 shadow / lighting | shadow points the wrong way for the visible light, detached or frozen shadow, shadow length or blur inconsistent with the light | soft or missing shadows under diffuse light, reflections |
| 2 mechanics | objects passing through each other, motion starting or changing with no contact, floating or falling up, impossible bounces, sudden stops or speed-ups while sliding | camera shake, motion blur, plausible but unusual motion (very bouncy or very heavy objects) |
| 3 causality | an effect that starts before its cause (a target moves before it is hit, a domino falls before it is touched) | the effect starting a few frames late |

If a clip shows several violations, label the first one and note the others in `notes`.
Rendering artefacts (texture flicker, morphing shapes, objects appearing from nowhere) go into
`artefact = 1` and are reported separately; they are not one of the three classes.

## 4. How onset and offset are marked

- Work frame by frame at the clip's native frame rate (convert timestamps to frames at 24 fps for the file).
- `onset_frame`: first frame in which the violation is visible. For class 3, the first frame in which the
  effect moves.
- `offset_frame`: first frame after which the violation is no longer visible (exclusive, like the synthetic
  labels). For class 3, the frame of the true contact plus one. Use the last frame + 1 if it lasts to the end.

## 5. File format

`labels.csv`, one row per clip:

```
clip_file,source,generator,class,onset_frame,offset_frame,fps,artefact,annotator_a,annotator_b,notes
v0001.mp4,ai,ModelX v2 (2026-09),2,31,52,24,0,ab,cd,"ball rises without contact"
v0002.mp4,real,own recording,0,,,30,0,ab,cd,
```

## 6. Checks before use

- No clip appears in two splits or twice in the set (check by perceptual hash).
- Class balance and source balance are reported with the results.
- Keep the raw annotations of both annotators alongside the resolved labels.
