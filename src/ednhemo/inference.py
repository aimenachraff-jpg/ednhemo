"""End-to-end inference: scan in, lesions out."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .lesions import Lesion, find_lesions
from .model import UNet2D
from .preprocessing import IMAGE_SIZE, Volume, load_volume, resize_slice, resize_to
from .weights import MODELS, load_model, resolve_device, resolve_model_name

__all__ = ["DetectionResult", "analyze", "predict_probabilities"]

DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_VOLUME_ML = 0.01


@dataclass
class DetectionResult:
    """Everything produced by :func:`analyze`.

    Attributes
    ----------
    lesions:
        Detected hemorrhages, sorted by slice then volume.
    mask:
        Binary mask ``(X, Y, Z)`` at the input resolution.
    probabilities:
        Sigmoid probabilities at the input resolution when
        ``keep_probabilities=True``, otherwise ``None``.
    spacing:
        Voxel size in millimetres.
    affine:
        NIfTI affine, or ``None`` for a 2D image.
    source, model, threshold, is_nifti:
        Provenance of the analysis.
    """

    lesions: list[Lesion]
    mask: np.ndarray
    spacing: tuple[float, float, float]
    affine: np.ndarray | None
    source: str
    model: str
    threshold: float
    is_nifti: bool
    probabilities: np.ndarray | None = None
    _figure_png: bytes | None = field(default=None, repr=False)

    @property
    def n_lesions(self) -> int:
        return len(self.lesions)

    @property
    def n_small_lesions(self) -> int:
        """Number of lesions smaller than 5 mL."""
        return sum(1 for lesion in self.lesions if lesion.volume_ml < 5.0)

    @property
    def total_volume_ml(self) -> float:
        return float(sum(lesion.volume_ml for lesion in self.lesions))

    @property
    def positive_voxels(self) -> int:
        return int(self.mask.sum())

    @property
    def n_slices(self) -> int:
        return int(self.mask.shape[2])

    def summary(self) -> dict[str, object]:
        """JSON-serializable summary of the analysis."""
        voxel_ml = float(np.prod(self.spacing)) / 1000.0
        return {
            "source": self.source,
            "model": self.model,
            "model_label": MODELS[self.model]["label"],
            "threshold": self.threshold,
            "slices": self.n_slices,
            "spacing_mm": [round(float(value), 6) for value in self.spacing],
            "voxel_ml": round(voxel_ml, 6),
            "positive_voxels": self.positive_voxels,
            "n_lesions": self.n_lesions,
            "n_under_5ml": self.n_small_lesions,
            "total_volume_ml": round(self.total_volume_ml, 3),
            "lesions": [lesion.as_dict() for lesion in self.lesions],
        }

    def save(self, out_dir: str | Path) -> list[Path]:
        """Write mask, lesion table, summary and overlay to ``out_dir``.

        Returns the list of files written.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        if self.affine is not None:
            import nibabel as nib

            mask_path = out / "mask.nii.gz"
            image = nib.Nifti1Image(self.mask.astype(np.uint8), self.affine)
            image.header.set_zooms(self.spacing)
            nib.save(image, str(mask_path))
        else:
            from PIL import Image

            mask_path = out / "mask.png"
            Image.fromarray(self.mask[..., 0].astype(np.uint8) * 255).save(mask_path)
        written.append(mask_path)

        csv_path = out / "lesions.csv"
        rows = [lesion.as_dict() for lesion in self.lesions]
        fieldnames = ["lesion", "slice", "volume_ml", "size_class", "voxels", "confidence", "x", "y", "z"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        written.append(csv_path)

        summary_path = out / "summary.json"
        summary_path.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(summary_path)

        if self._figure_png is not None:
            overlay_path = out / "overlay.png"
            overlay_path.write_bytes(self._figure_png)
            written.append(overlay_path)

        return written

    def __repr__(self) -> str:
        return (
            f"DetectionResult(n_lesions={self.n_lesions}, "
            f"under_5ml={self.n_small_lesions}, model={self.model!r}, "
            f"volume_ml={self.total_volume_ml:.3f})"
        )


def predict_probabilities(
    volume: Volume,
    model: UNet2D | None = None,
    *,
    model_name: str = "variante",
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Run the U-Net slice by slice and return probabilities at input resolution."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    target = resolve_device(device)
    net = model if model is not None else load_model(model_name, device=target)
    net.eval()

    slices = [resize_slice(volume.data[..., z]) for z in range(volume.n_slices)]
    probs_512 = np.empty((IMAGE_SIZE, IMAGE_SIZE, volume.n_slices), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(slices), batch_size):
            batch = np.stack(slices[start : start + batch_size])
            tensor = torch.from_numpy(batch[:, None]).to(target)
            logits = net(tensor)
            probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
            probs_512[:, :, start : start + probs.shape[0]] = probs.transpose(1, 2, 0)

    height, width = volume.data.shape[:2]
    if (height, width) == (IMAGE_SIZE, IMAGE_SIZE):
        return probs_512

    probs = np.empty((height, width, volume.n_slices), dtype=np.float32)
    for z in range(volume.n_slices):
        probs[..., z] = resize_to(probs_512[..., z], (height, width), order=1)
    return probs


def analyze(
    source: str | Path | np.ndarray,
    model: str = "variante",
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_volume_ml: float = DEFAULT_MIN_VOLUME_ML,
    figure: bool = True,
    keep_probabilities: bool = False,
) -> DetectionResult:
    """Detect intracranial hemorrhages in a CT scan or a 2D slice.

    Parameters
    ----------
    source:
        NIfTI path (``.nii`` / ``.nii.gz``), 2D image path (already windowed),
        or a raw HU array of shape ``(X, Y)`` / ``(X, Y, Z)``.
    model:
        ``"socle"`` (reference) or ``"variante"`` (small-lesion oversampling,
        the default).
    threshold:
        Probability threshold, ``0.5`` as in the official evaluation.
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``.
    batch_size:
        Number of slices per forward pass.
    min_volume_ml:
        Components below this volume are ignored (0.01 mL as in the project).
    figure:
        Render a PNG overlay (stored in the result and written by ``save``).
    keep_probabilities:
        Keep the full-resolution probability map in the result.

    Returns
    -------
    DetectionResult
        Lesions, binary mask, spacing and metadata.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    model_name = resolve_model_name(model)
    volume = load_volume(source)
    probabilities = predict_probabilities(
        volume,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    mask = probabilities > threshold
    lesions = find_lesions(mask, probabilities, volume.spacing, min_volume_ml=min_volume_ml)

    figure_png: bytes | None = None
    if figure:
        from .viz import render_overlay

        figure_png = render_overlay(
            volume.data,
            mask,
            lesions,
            model_name=model_name,
            threshold=threshold,
        )

    result = DetectionResult(
        lesions=lesions,
        mask=mask,
        spacing=volume.spacing,
        affine=volume.affine,
        source=volume.source,
        model=model_name,
        threshold=threshold,
        is_nifti=volume.is_nifti,
        probabilities=probabilities if keep_probabilities else None,
        _figure_png=figure_png,
    )
    return result
