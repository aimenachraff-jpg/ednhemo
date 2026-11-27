"""Input loading and the exact training-time normalization.

Everything here mirrors the training pipeline:

* CT values are clipped to the brain window ``[-40, 120]`` HU,
* then normalized with ``(value - 40) / 80``, which maps the window to
  ``[-1, +1]``,
* slices are resized to ``512 x 512`` with bilinear interpolation.

A 2D image (PNG/JPEG) is assumed to be already windowed: 8-bit values are
mapped to ``[-1, +1]`` with ``2 * (v / 255) - 1``, the exact equivalent of the
HU normalization for a window of 160 HU centered on 40 HU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

__all__ = [
    "CT_MIN_HU",
    "CT_MAX_HU",
    "IMAGE_SIZE",
    "BHSD_SPACING",
    "Volume",
    "load_volume",
    "normalize_hu",
    "normalize_windowed",
    "resize_slice",
    "resize_to",
]

CT_MIN_HU = -40.0
CT_MAX_HU = 120.0
IMAGE_SIZE = 512

#: Typical BHSD voxel size in millimetres, ``(x, y, z)``. Used when a source
#: does not carry spatial metadata (2D images).
BHSD_SPACING: tuple[float, float, float] = (0.48828125, 0.48828125, 6.0)

_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class Volume:
    """A normalized CT volume ready for inference.

    Attributes
    ----------
    data:
        ``(X, Y, Z)`` float32 array already windowed and normalized to
        ``[-1, 1]``.
    spacing:
        Voxel size in millimetres, ``(x, y, z)``.
    affine:
        NIfTI affine when the input was a NIfTI file, ``None`` for 2D images.
    source:
        Human-readable origin of the data.
    is_nifti:
        ``True`` when the input came from a NIfTI file.
    """

    data: np.ndarray
    spacing: tuple[float, float, float]
    affine: np.ndarray | None
    source: str
    is_nifti: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def n_slices(self) -> int:
        return int(self.data.shape[2])


def normalize_hu(values: np.ndarray) -> np.ndarray:
    """Clip to ``[-40, 120]`` HU and map to ``[-1, +1]``."""
    return (np.clip(np.asarray(values, dtype=np.float32), CT_MIN_HU, CT_MAX_HU) - 40.0) / 80.0


def normalize_windowed(values: np.ndarray) -> np.ndarray:
    """Map an already windowed 8-bit image (0-255) to ``[-1, +1]``."""
    values = np.asarray(values, dtype=np.float32)
    if values.size and float(values.max()) > 1.5:
        values = values / 255.0
    return 2.0 * values - 1.0


def _as_3d(array: np.ndarray, source: str) -> np.ndarray:
    if array.ndim == 2:
        return array[..., None]
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(
        f"{source}: expected a 2D or 3D volume, got shape {array.shape}. "
        "4D volumes with several channels are not supported."
    )


def _clean_spacing(zooms: tuple[float, ...]) -> tuple[float, float, float]:
    spacing = []
    for value, fallback in zip(zooms, BHSD_SPACING):
        value = float(value) if value is not None else 0.0
        spacing.append(value if np.isfinite(value) and value > 0 else fallback)
    return (spacing[0], spacing[1], spacing[2])


def load_volume(source: str | Path | np.ndarray) -> Volume:
    """Load a NIfTI scan or a 2D image and normalize it.

    Parameters
    ----------
    source:
        Path to ``.nii`` / ``.nii.gz`` / ``.png`` / ``.jpg`` / ``.tif``, or a
        raw HU array of shape ``(X, Y)`` or ``(X, Y, Z)``.
    """
    if isinstance(source, np.ndarray):
        array = _as_3d(np.asarray(source, dtype=np.float32), "<array>")
        return Volume(
            data=normalize_hu(array),
            spacing=BHSD_SPACING,
            affine=None,
            source="<array>",
            is_nifti=False,
        )

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    name = path.name.lower()

    if name.endswith(_NIFTI_SUFFIXES):
        image = nib.load(str(path))
        array = _as_3d(np.asanyarray(image.dataobj, dtype=np.float32), path.name)
        zooms = image.header.get_zooms()[:3]
        return Volume(
            data=normalize_hu(array),
            spacing=_clean_spacing(zooms),
            affine=np.asarray(image.affine, dtype=np.float64),
            source=str(path),
            is_nifti=True,
        )

    if name.endswith(_IMAGE_SUFFIXES):
        with Image.open(path) as handle:
            grayscale = handle.convert("L")
            array = np.asarray(grayscale, dtype=np.float32)
        return Volume(
            data=normalize_windowed(array)[..., None],
            spacing=BHSD_SPACING,
            affine=None,
            source=str(path),
            is_nifti=False,
        )

    raise ValueError(
        f"Unsupported format: {path.suffix or path.name}. "
        "Use NIfTI (.nii, .nii.gz) or a 2D image (.png, .jpg, .tif)."
    )


def _fit_shape(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crop or edge-pad ``array`` so it has exactly ``shape``."""
    if array.shape == shape:
        return array
    h, w = shape
    out = array[:h, :w]
    if out.shape[0] < h or out.shape[1] < w:
        pad = ((0, max(0, h - out.shape[0])), (0, max(0, w - out.shape[1])))
        out = np.pad(out, pad, mode="edge")
    return out


def resize_to(
    array: np.ndarray,
    shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Resize a 2D array to ``shape`` (bilinear by default)."""
    array = np.asarray(array, dtype=np.float32)
    if array.shape == shape:
        return array
    factors = (shape[0] / array.shape[0], shape[1] / array.shape[1])
    resized = ndimage.zoom(array, factors, order=order)
    return _fit_shape(resized, shape)


def resize_slice(slice_2d: np.ndarray, size: int = IMAGE_SIZE, order: int = 1) -> np.ndarray:
    """Resize a 2D slice to ``size x size`` (bilinear by default)."""
    return resize_to(slice_2d, (size, size), order=order)
