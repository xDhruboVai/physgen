# Roadmap and milestones

The milestones are the ones from the design document (section 12). Status is as of 25 October 2026.

| Milestone | Scope | Exit criteria | Status |
|---|---|---|---|
| M0 | Decide D1 to D11; repository and asset manifest | choices recorded | **Done.** All eleven decisions use the recommended option; every alternative is a config switch. Private repository and versioned asset manifest in place. |
| M1 | Pilot: static camera, primitives, 2 to 3 scenarios, ~5k pairs | pipeline runs end to end; V1 to V10 pass; probes P1 to P6 near chance | **Done.** 4,934 training pairs (7,360 in total), validation clean, compliant-vs-violating probes at chance, first baseline trained and evaluated. |
| M2 | All scenarios, all subtypes including c3, severity levels | balance audit passes; first baseline results | **Code ready, data not yet generated.** All 7 scenarios and 18 subtypes work in test runs. Subtype balancing has to be fixed first. |
| M3 | YCB objects, moving cameras, Blender subset | renderer-transfer test runs | **Mostly ready.** YCB meshes and moving cameras are implemented and tested; the Blender renderer works and a 30-scene transfer test was run on the pilot. |
| M4 | ~20k pairs, OOD splits, real test set | full evaluation table | **Partly ready.** OOD splits are implemented and already evaluated on the pilot. The real / AI-generated test set has a drafted annotation protocol but no videos yet. |
| M5 | Dataset card and documentation | reproducible release | **Started.** Pilot dataset card written; the final card follows M4. |

## What happens next (proposed)

The pilot was produced on a laptop-class GPU (GTX 1050 Ti, 4 GB). The next steps move to a desktop
with an RTX 3060 Ti or 4060 Ti (8 GB), which makes encoder fine-tuning and the doc's full settings
practical. Durations are my estimates and assume the work runs largely unattended.

| Week | Work | Result |
|---|---|---|
| 1 | Set up the new machine, regenerate the pilot from the same seeds, benchmark; features at the doc's stride of 8; three seeds per baseline | pilot results with error bars |
| 2 | Fine-tune the encoder end to end (frozen vs fine-tuned ablation) | answer to whether the frozen encoder is the bottleneck |
| 3 | Balance the subtypes; generate M2/M3: all 7 scenarios, 40% YCB objects, 20% moving cameras, ~20k training pairs, 200 Blender scenes | M2 dataset, validated |
| 4 | Shortcut audit with stronger probes; baselines and ablations on M2 | M2 results table |
| 5 to 6 | Collect and annotate the real / AI-generated test set (two annotators, protocol already drafted) | test_real split and agreement statistics |
| 7 | Final evaluation table, dataset card, documentation | M4 and M5 complete |

## Risks and how I plan to handle them

- **The frozen encoder plateaus early** (validation stops improving after about 30 epochs). If
  fine-tuning in week 2 does not help, I will try a different backbone; the pipeline already accepts any
  video model.
- **Rare subtypes** (mid-air stop, super-elastic bounce, swapped masses) are 4 to 5 times less frequent
  than the common ones. Fix: per-subtype quotas in the sampler before M2.
- **Swapped masses (S2.6) may not be learnable** when no earlier collision shows the masses. I will keep
  it, report it separately, and discuss it as a limitation unless you prefer to drop it.
- **The real test set depends on people**, not compute: finding suitable clips and a second annotator
  is the part most likely to slip.

## Questions for you

1. Should the four-class results be reported per scenario as well? c3 only exists where there is a
   cause and an effect, so the scenario carries some information about the class.
2. Is it acceptable to use AI-generated videos from commercial models in the real test set, provided
   I record the model, version and prompt for each clip?
3. Who could act as the second annotator for the real test set?
4. Are you happy with the ~20k-pair target for M2, or should I aim higher if the learning curve is still
   rising (the design document's option C, 50k+)?
