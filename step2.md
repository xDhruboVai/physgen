# Step 2: continue the thesis dataset work on the new PC

Instructions for Claude Code. The user cloned this repo on a PC with an RTX 3060 Ti or 4060 Ti
(8 GB VRAM) and gave you the design doc PDF ("Physics-Violation Dataset Generator: Design and Data
Documentation"). Work through the stages below without waiting for the user, except where it says
to ask. Read README.md, results/README.md and docs/pilot/dataset_card.md first.

## Where things stand

- Milestone M1 (pilot) was done on a GTX 1050 Ti: 10,290 clips, `validate` 0 errors, shortcut probes
  at chance for c0 vs violating, frozen VideoMAE V2 + temporal head gives test_id AUROC 0.70-0.75 and
  segment F1@0.5 up to 0.38 (results/README.md).
- The pilot data (~15 GB) is not in git. You regenerate it here (stage 2): scenes are a pure function
  of (seed, generator code, asset manifest), so the same scenes and labels come back.
- Two things were scaled down for the old GPU and must be redone here: feature stride 16 instead of
  the doc's 8, and no encoder fine-tuning.

## Rules

- Commit as the user only (their git identity). Never add a "Co-Authored-By: Claude" line. Push to
  `origin main` after each stage, with a clear message.
- Python packages only in the project venv `.venv`; tools such as Blender go in `third_party/`.
  Large outputs stay out of git (`datasets/`, `*.pt`, predictions; see .gitignore).
- Never execute Hugging Face repository code (no `trust_remote_code`). VideoMAE V2 is loaded by
  `physgen.features.load_videomaev2`, which maps its weights onto the built-in VideoMAE
  (`tests/test_videomaev2.py`).
- Don't change defaults taken from the design doc; add alternatives as switches in
  `physgen/default.yaml` (or a file in `configs/`).
- Generate data only from a clean, pushed commit: every clip records the generator commit.
- Run `tests/test_pipeline.py`, `tests/test_baseline.py` and `tests/test_videomaev2.py` before each push.
- Long jobs run in the background. `generate` and `export-features` resume when rerun after an
  interruption, so after a crash or reboot just rerun the same command.
- Check GPU use with `nvidia-smi` (Task Manager's "3D" graph does not show CUDA work). Feature export
  and training should keep it near 100%.
- Ask the user before deleting anything you did not create, or before a single step that would take
  more than about 24 hours of compute.
- If the PDF cannot be read directly, `pip install pypdf` into `.venv` and extract its text.

Commands below use Windows paths (`.venv/Scripts/python`); on Linux use `.venv/bin/python`.

## Stage 1: setup

1. Follow README "Setup on a new machine": venv with Python 3.11, torch from the cu128 index,
   `pip install -r requirements.txt`, pybullet (Windows: conda-forge files as described there), and
   Blender 4.2 LTS portable unpacked into `third_party/blender-4.2.23-windows-x64/` (or set
   `renderer.blender.executable`). Blender only uses OptiX on RTX cards; keep `device: OPTIX`.
2. `.venv/Scripts/python -m physgen fetch-assets` restores exactly the assets listed in
   `assets/manifest.json` (Poly Haven textures/HDRIs, procedural patterns, YCB meshes).
3. Run the three test files, then `tests/benchmark.py`. Put this GPU's numbers in the README
   performance table (add a column; keep the GTX 1050 Ti numbers).

## Stage 2: reproduce the pilot (M1)

Same config and seeds as on the old PC (`configs/pilot.yaml`):

```bash
P=.venv/Scripts/python; R=datasets/pilot
$P -m physgen generate --config configs/pilot.yaml --split train             --num-scenes 1900 --out $R
$P -m physgen generate --config configs/pilot.yaml --split val               --num-scenes 200  --out $R
$P -m physgen generate --config configs/pilot.yaml --split test_id           --num-scenes 200  --out $R
$P -m physgen generate --config configs/pilot.yaml --split test_ood_objects  --num-scenes 150  --out $R
$P -m physgen generate --config configs/pilot.yaml --split test_ood_env      --num-scenes 150  --out $R
$P -m physgen generate --config configs/pilot.yaml --split test_ood_scenario --num-scenes 150  --out $R
$P -m physgen generate --config configs/pilot.yaml --split test_ood_subtype  --num-scenes 150  --out $R
$P -m physgen generate --config configs/pilot.yaml --split test_renderer --renderer blender --num-scenes 30 --workers 1 --out $R
$P -m physgen validate --root $R
$P -m physgen audit --root $R
```

Use `--workers` = CPU cores minus one for the raster splits. Then check the reproduction:
- `validate` must report 0 errors.
- Clips per split and class should match `docs/pilot/dataset_card.md`. Small differences (about 1%)
  are acceptable: whether a subtle violation passes the visibility check (V6) depends on rendered
  pixels, which can differ slightly across GPUs and drivers. Report the differences.
- Probe results should be close to `docs/pilot/probes.csv`.

## Stage 3: pilot baseline at the doc's settings

1. `export-features --root datasets/pilot --stride 8` (VideoMAE V2-Base, the default model). Use the
   largest `--batch` that fits in 8 GB.
2. Train and evaluate the head without and with `--pairwise`, 3 seeds each (`--seed 0/1/2`), output
   folders under `results/pilot_s8_*`. Report mean and std per split, and compare with the stride-16
   numbers in results/README.md.

## Stage 4: fine-tuning ablation (doc section 10: frozen vs fine-tuned encoder)

Implement end-to-end fine-tuning of VideoMAE V2 in `baseline/` (new module, same metrics from
`baseline/metrics.py`, same splits):
- Load with `load_videomaev2`, read frames from the WebDataset shards (`physgen.writer.read_shard`,
  `decode`), 16-frame windows at stride 8, the same temporal head on top.
- Fit in 8 GB: bf16/fp16 autocast, gradient checkpointing, small batch with gradient accumulation;
  if still too large, unfreeze only the last N blocks (make N an option and report what was used).
- Train on the pilot train split, tune on val, evaluate on every split. Add a small test.
Report frozen vs fine-tuned side by side.

## Stage 5: generator improvements before scaling up

Each as a config option with a test:
- Subtype balance (validator V10 only reports): S2.4, S2.5, S2.6 and S1.6 come out 4-5x rarer than
  the common subtypes because they are often infeasible. Add quota-based rebalancing so per-class
  subtype counts stay within about 2x of each other.
- In the pilot, test_ood_subtype had no c3 because the stack scenario (held-out S3.3) was disabled.
  With configs/full.yaml it is enabled; confirm every OOD split has all three classes where the
  design allows.

## Stage 6: milestone M2

`configs/full.yaml`: all 7 scenarios, YCB meshes 40%, 20% moving cameras. First preview a few scenes
per scenario (`python -m physgen preview --config configs/full.yaml --scenario <name> --index <i>`) and
look at the PNG contact sheets; with moving cameras check that objects stay readable and in view.
Then generate into `datasets/m2`: train ~6,500 scenes (about 20k pairs), val 500, test_id 500, each
OOD split 300, test_renderer 200 (Blender). Run `validate` (0 errors), `audit` (c0-vs-violating
probes near 0.5; if a probe finds a shortcut, fix the generator and regenerate), and
`python -m physgen card --root datasets/m2`; copy the card, `stats/probes.csv`, `stats/balance.md`,
the config and `splits/heldout.json` into `docs/m2/`.

## Stage 7: M2 baselines and write-up

Features at stride 8, frozen head (independent and pairwise, 3 seeds) and the fine-tuned model from
stage 4, all evaluated on every M2 split. Write `results/README.md` sections for the pilot at stride 8,
the fine-tuning ablation and M2: tables per split, breakdowns by class, subtype, severity, scenario
and camera mode, and short observations (what is hard, OOD and renderer gaps, effect of each ablation).
Update the README "Pilot results" section into a general results summary. Push.

## When you finish

Tell the user what was done, the headline numbers, anything that failed or was skipped, and what is
left (the real / AI-generated test set needs videos collected by hand: docs/real_test_protocol.md).
