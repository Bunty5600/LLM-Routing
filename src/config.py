"""Loads configs/experiment.yaml and resolves paths from the repo root."""
from pathlib import Path

import yaml  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

ROOT = Path(__file__).resolve().parent.parent


def load_config(path="configs/experiment.yaml"):
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str):
    """Turn a repo-relative path from the config into an absolute Path."""
    return ROOT / path_str