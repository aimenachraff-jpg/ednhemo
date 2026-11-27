"""Checkpoint registry and loading.

Two checkpoints are shipped inside the wheel, so ``pip install ednhemo`` works
offline:

* ``socle``    - the reference model trained with the common protocol,
* ``variante`` - the lot G model trained with 4 small-hemorrhage slices
  guaranteed in every batch of 8. Better lesion sensitivity, slightly more
  false alarms.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

import torch

from .model import UNet2D, count_parameters

__all__ = ["MODELS", "MODEL_ALIASES", "ModelName", "resolve_model_name", "load_model", "list_models"]

ModelName = Literal["socle", "variante"]

MODELS: dict[str, dict[str, object]] = {
    "socle": {
        "file": "model_socle.pt",
        "label": "Reference model (common protocol)",
        "dice": 0.5098,
        "sensitivity": 41.9,
        "false_alarms_per_scan": 3.43,
        "validation_dice": 0.5845,
        "best_epoch": 26,
    },
    "variante": {
        "file": "model_variante.pt",
        "label": "Lot G model (small-lesion oversampling)",
        "dice": 0.5399,
        "sensitivity": 44.1,
        "false_alarms_per_scan": 3.78,
        "validation_dice": 0.5544,
        "best_epoch": 35,
    },
}

MODEL_ALIASES: dict[str, str] = {
    "base": "socle",
    "reference": "socle",
    "standard": "socle",
    "small": "variante",
    "small-lesions": "variante",
    "lot-g": "variante",
    "lotg": "variante",
}

_CACHE: dict[tuple[str, str], UNet2D] = {}


def resolve_model_name(name: str) -> str:
    """Resolve aliases and validate a checkpoint name."""
    key = str(name).strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in MODELS:
        available = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}.")
    return key


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Turn ``"auto"`` into CUDA when available, CPU otherwise."""
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(
    name: str = "variante",
    device: str | torch.device | None = "auto",
) -> UNet2D:
    """Load a published checkpoint in evaluation mode.

    Parameters
    ----------
    name:
        ``"socle"`` or ``"variante"`` (aliases such as ``"base"`` and
        ``"small"`` are accepted).
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"`` or any ``torch.device``.

    Returns
    -------
    UNet2D
        Model with its weights loaded and ``eval()`` already called.
    """
    key = resolve_model_name(name)
    target = resolve_device(device)
    cache_key = (key, str(target))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    checkpoint = files("ednhemo").joinpath("weights", str(MODELS[key]["file"]))
    with checkpoint.open("rb") as handle:
        try:
            state = torch.load(handle, map_location=target, weights_only=True)
        except TypeError:  # torch < 2.0
            handle.seek(0)
            state = torch.load(handle, map_location=target)

    model = UNet2D().to(target)
    model.load_state_dict(state)
    model.eval()
    _CACHE[cache_key] = model
    return model


def list_models() -> list[dict[str, object]]:
    """Return a description of every shipped checkpoint."""
    rows = []
    for name, info in MODELS.items():
        rows.append({"name": name, **info})
    return rows


def checkpoint_path(name: str) -> Path:
    """Filesystem path of a shipped checkpoint (useful for inspection)."""
    key = resolve_model_name(name)
    return Path(str(files("ednhemo").joinpath("weights", str(MODELS[key]["file"]))))


def model_size(name: str = "variante") -> int:
    """Number of parameters of a checkpoint."""
    return count_parameters(load_model(name, device="cpu"))
