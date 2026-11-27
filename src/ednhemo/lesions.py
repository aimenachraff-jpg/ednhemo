"""Lesion extraction: connected components, volumes and size classes.

A lesion is a 26-connected component of the binary mask. Components smaller
than ``min_volume_ml`` (0.01 mL by default, as in the project) are discarded.
This is exactly the lesion-by-lesion view used by ``evaluation_commune.py``:
one missed 30-voxel lesion matters as much as one missed 30,000-voxel lesion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "CONNECTIVITY_26",
    "SIZE_CLASSES",
    "Lesion",
    "size_class",
    "find_lesions",
]

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)

#: ``(upper bound in mL, label)``; the first class is ``< 0.1 mL``.
SIZE_CLASSES: tuple[tuple[float, str], ...] = (
    (0.1, "< 0.1 mL"),
    (1.0, "0.1-1 mL"),
    (5.0, "1-5 mL"),
    (float("inf"), "> 5 mL"),
)


@dataclass(frozen=True)
class Lesion:
    """One connected hemorrhagic component."""

    index: int
    slice: int
    volume_ml: float
    size_class: str
    voxels: int
    confidence: float
    centroid: tuple[float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "lesion": self.index,
            "slice": self.slice,
            "volume_ml": round(self.volume_ml, 3),
            "size_class": self.size_class,
            "voxels": self.voxels,
            "confidence": round(self.confidence, 3),
            "x": round(self.centroid[0], 1),
            "y": round(self.centroid[1], 1),
            "z": round(self.centroid[2], 1),
        }


def size_class(volume_ml: float) -> str:
    """Return the size class label for a volume in millilitres."""
    for upper, label in SIZE_CLASSES:
        if volume_ml < upper:
            return label
    return SIZE_CLASSES[-1][1]


def find_lesions(
    mask: np.ndarray,
    probabilities: np.ndarray,
    spacing: tuple[float, float, float],
    min_volume_ml: float = 0.01,
) -> list[Lesion]:
    """Label a binary mask and describe every component.

    Parameters
    ----------
    mask:
        Boolean ``(X, Y, Z)`` array.
    probabilities:
        Voxel-wise sigmoid probabilities, same shape as ``mask``. Used to
        report a mean confidence per lesion.
    spacing:
        Voxel size in millimetres, ``(x, y, z)``.
    min_volume_ml:
        Components below this volume are ignored.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask.shape}")

    labels, n_labels = ndimage.label(mask, structure=CONNECTIVITY_26)
    if n_labels == 0:
        return []

    voxel_ml = float(np.prod(spacing)) / 1000.0
    counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    sums = ndimage.sum(np.asarray(probabilities, dtype=np.float32), labels, index=np.arange(1, n_labels + 1))
    centroids = ndimage.center_of_mass(mask, labels, index=np.arange(1, n_labels + 1))

    lesions: list[Lesion] = []
    for label_id in range(1, n_labels + 1):
        voxels = int(counts[label_id])
        volume = voxels * voxel_ml
        if volume < min_volume_ml:
            continue
        centroid = centroids[label_id - 1]
        lesions.append(
            Lesion(
                index=0,  # assigned after sorting
                slice=int(round(float(centroid[2]))),
                volume_ml=volume,
                size_class=size_class(volume),
                voxels=voxels,
                confidence=float(sums[label_id - 1]) / voxels,
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            )
        )

    lesions.sort(key=lambda item: (item.slice, -item.volume_ml))
    return [
        Lesion(
            index=i,
            slice=item.slice,
            volume_ml=item.volume_ml,
            size_class=item.size_class,
            voxels=item.voxels,
            confidence=item.confidence,
            centroid=item.centroid,
        )
        for i, item in enumerate(lesions, start=1)
    ]
