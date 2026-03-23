from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    return os.path.relpath(target, repo_root())
