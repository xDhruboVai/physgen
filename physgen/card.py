"""Dataset card (doc milestone M5): dataset_root/README.md built from the manifests, configs and stats."""
import glob
import json
import os

import pandas as pd

from . import writer


def _table(df):
    return df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()


def write_card(root):
    clips, pairs = writer.rebuild_manifests(root)
    if clips is None:
        raise SystemExit(f"no manifests under {root}")
    cfgs = {}
    for p in sorted(glob.glob(os.path.join(root, "config-*.json"))):
        with open(p) as f:
            cfgs[os.path.basename(p)[7:-5]] = json.load(f)
    any_cfg = next(iter(cfgs.values()), {})
    clip = any_cfg.get("dataset", {}).get("clip", {})
    commits = sorted(clips["generator_commit"].dropna().unique()) if "generator_commit" in clips else []
    manifests = sorted(clips["asset_manifest"].unique())
    held = {}
    if os.path.exists(os.path.join(root, "splits", "heldout.json")):
        with open(os.path.join(root, "splits", "heldout.json")) as f:
            held = json.load(f)
    probes = os.path.join(root, "stats", "probes.csv")
    md = [
        "# Physics-violation video dataset",
        "",
        "Synthetic minimal pairs for detecting and temporally localising physics violations. Every scene",
        "yields a physics-compliant clip (class 0) and up to three violating clips from the same seed:",
        "1 shadow/lighting, 2 rigid-body mechanics, 3 causality (effect before cause).",
        "Frames before the labelled onset are identical to the compliant clip.",
        "",
        "## Provenance",
        "",
        f"- Generator: physgen {clips['generator_version'].iloc[0]}, commit(s) {', '.join(commits) or 'unknown'}",
        f"- Asset manifest hash(es): {', '.join(manifests)}",
        f"- Renderer(s): {', '.join(sorted(clips['renderer'].unique()))}",
        f"- Clip format: {clip.get('frames')} frames at {clip.get('fps')} fps, {clip.get('resolution')} px, "
        f"{any_cfg.get('output', {}).get('format', '?')} frames in WebDataset tar shards",
        "- Every clip can be regenerated from its scene config in `configs/<split>/` with the recorded commit.",
        "",
        "## Contents",
        "",
        _table(clips.groupby(["split", "class"]).size().unstack(fill_value=0)),
        "",
        "Pairs per subtype and severity:",
        "",
        _table(pairs.groupby(["subtype", "severity"]).size().unstack(fill_value=0)),
        "",
        "Pairs per scenario and class:",
        "",
        _table(pairs.groupby(["scenario", "class"]).size().unstack(fill_value=0)),
        "",
        "## Splits",
        "",
        "Split by scene seed; held-out assets (~15% of textures, HDRIs and meshes), shape families, scenarios",
        "and subtypes appear only in their OOD split. `test_renderer` re-renders `test_id` scenes with another",
        "renderer (ids carry the renderer name).",
        "",
        f"- Held-out shape families: {held.get('heldout_shapes')}",
        f"- Held-out scenarios: {held.get('heldout_scenarios')}",
        f"- Held-out subtypes: {held.get('heldout_subtypes')}",
        f"- Held-out asset ids: `splits/heldout.json` ({len(held.get('assets', []))} assets)",
        "",
        "## Labels and files",
        "",
        "- `shards/<split>/*.tar`: `<clip_id>.<frame>.jpg` plus `<clip_id>.json` (all metadata).",
        "- `manifest/clips.parquet` (one row per clip), `manifest/pairs.parquet` (one row per pair).",
        "- `onset_frame`, `offset_frame`: half-open interval `[onset, offset)`; `frame_labels` (0/1) and",
        "  `frame_classes` (0-3) per frame; `subtype`, `severity` (subtle/clear/strong), `timing`",
        "  (persistent/transient/event), `involved_objects`, `violation_params`.",
        "",
    ]
    if os.path.exists(probes):
        md += ["## Shortcut probes (balanced accuracy; should stay near chance)", "",
               _table(pd.read_csv(probes).set_index("probe")), ""]
    md += [
        "## Licences",
        "",
        "- Textures and HDRIs: Poly Haven, CC0. Procedural patterns: generated, CC0.",
        "- YCB object meshes (if used): Calli et al., *The YCB Object and Model Set*, CC BY 4.0 - attribution required.",
        "- Frozen-feature exports from VideoMAE V2 weights inherit their CC BY-NC 4.0 licence.",
        "",
        "## Known limitations",
        "",
        "- Mass inconsistency (S2.6) is only detectable when masses were observable earlier in the clip.",
        "- In c3 the cause object still reacts at the true contact time (design caveat, doc section 6.3).",
        "- c3 exists only in scenarios with a cause-effect interaction, so scenario and class correlate.",
    ]
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("dataset card ->", os.path.join(root, "README.md"))
