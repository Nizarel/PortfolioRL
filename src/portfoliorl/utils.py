"""Reproducibility helpers: seeding and lightweight run bookkeeping."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    ``torch`` is imported lazily so that the data and metrics modules remain
    usable in environments where PyTorch is unavailable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(False)  # CPU MLP is already stable
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` to ``path`` as indented JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
