"""Overlay rendering (headless, no GUI required)."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .lesions import Lesion

__all__ = ["render_overlay"]


def _slices_to_show(lesions: list[Lesion], n_slices: int, max_panels: int = 4) -> list[int]:
    if not lesions:
        return [n_slices // 2]
    totals: dict[int, float] = {}
    for lesion in lesions:
        totals[lesion.slice] = totals.get(lesion.slice, 0.0) + lesion.volume_ml
    ranked = sorted(totals, key=lambda z: totals[z], reverse=True)[:max_panels]
    return sorted(ranked)


def render_overlay(
    volume: np.ndarray,
    mask: np.ndarray,
    lesions: list[Lesion],
    *,
    model_name: str = "variante",
    threshold: float = 0.5,
) -> bytes:
    """Render up to four slices with the predicted contour and return PNG bytes."""
    slices = _slices_to_show(lesions, int(mask.shape[2]))
    n_panels = len(slices)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5), squeeze=False)

    for ax, z in zip(axes[0], slices):
        ax.imshow(volume[:, :, z], cmap="gray", origin="lower", vmin=-1.0, vmax=1.0)
        mask_2d = mask[:, :, z]
        if mask_2d.any():
            ax.contour(mask_2d, levels=[0.5], colors="#d62728", linewidths=1.4)
            ax.imshow(
                np.ma.masked_where(~mask_2d, mask_2d),
                cmap="Reds",
                alpha=0.35,
                origin="lower",
            )
        volumes = [lesion.volume_ml for lesion in lesions if lesion.slice == z]
        if volumes:
            title = f"slice {z} - {sum(volumes):.2f} mL"
        else:
            title = f"slice {z} - no lesion"
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"ednhemo - {model_name} - threshold {threshold:.2f}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()
