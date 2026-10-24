import copy
import os

import yaml

DEFAULT = os.path.join(os.path.dirname(__file__), "default.yaml")


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path=None, overrides=None):
    with open(DEFAULT) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path, encoding="utf-8-sig") as f:  # Windows editors often write a BOM
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(cfg)
        if unknown:
            raise KeyError(f"unknown config section(s) {sorted(unknown)} in {path}")
        cfg = merge(cfg, user)
    return merge(cfg, overrides)
