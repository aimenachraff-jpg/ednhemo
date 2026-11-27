"""Lesion extraction: components, volumes, size classes and filtering."""

import numpy as np
import pytest

from ednhemo import find_lesions, size_class


def _blob(shape, slices, rows, cols):
    mask = np.zeros(shape, dtype=bool)
    mask[rows[0] : rows[1], cols[0] : cols[1], slices[0] : slices[1]] = True
    return mask


def test_size_class_boundaries():
    assert size_class(0.0) == "< 0.1 mL"
    assert size_class(0.099) == "< 0.1 mL"
    assert size_class(0.1) == "0.1-1 mL"
    assert size_class(0.999) == "0.1-1 mL"
    assert size_class(1.0) == "1-5 mL"
    assert size_class(4.999) == "1-5 mL"
    assert size_class(5.0) == "> 5 mL"
    assert size_class(120.0) == "> 5 mL"


def test_two_components_are_counted_separately():
    mask = _blob((20, 20, 5), (2, 3), (2, 6), (2, 6))  # 4 x 4 x 1 = 16 voxels
    mask[10:14, 10:14, 2] = True  # 16 voxels
    probabilities = np.where(mask, 0.9, 0.1)

    lesions = find_lesions(mask, probabilities, spacing=(1.0, 1.0, 1.0))

    assert len(lesions) == 2
    assert [lesion.voxels for lesion in lesions] == [16, 16]
    assert all(lesion.volume_ml == 0.016 for lesion in lesions)


def test_volumes_use_the_spacing():
    mask = _blob((10, 10, 2), (0, 1), (0, 2), (0, 2))  # 4 voxels
    probabilities = np.where(mask, 0.8, 0.0)

    lesions = find_lesions(mask, probabilities, spacing=(0.5, 0.5, 2.0), min_volume_ml=0.0)

    assert len(lesions) == 1
    assert lesions[0].volume_ml == pytest.approx(4 * 0.5 * 0.5 * 2.0 / 1000.0)
    assert lesions[0].confidence == pytest.approx(0.8)


def test_small_components_are_discarded():
    mask = np.zeros((10, 10, 2), dtype=bool)
    mask[1, 1, 1] = True  # 1 mm3 = 0.001 mL, below the 0.01 mL floor
    probabilities = np.where(mask, 1.0, 0.0)

    assert find_lesions(mask, probabilities, spacing=(1.0, 1.0, 1.0)) == []


def test_empty_mask_returns_no_lesion():
    mask = np.zeros((8, 8, 3), dtype=bool)
    assert find_lesions(mask, np.zeros_like(mask, dtype=np.float32), (1.0, 1.0, 1.0)) == []


def test_lesions_are_sorted_by_slice_then_volume():
    mask = np.zeros((20, 20, 4), dtype=bool)
    mask[2:5, 2:6, 3] = True  # slice 3, 12 voxels
    mask[10:13, 10:14, 1] = True  # slice 1, 12 voxels
    probabilities = np.full(mask.shape, 0.75, dtype=np.float32)

    lesions = find_lesions(mask, probabilities, spacing=(1.0, 1.0, 1.0))

    assert [lesion.slice for lesion in lesions] == [1, 3]
    assert [lesion.index for lesion in lesions] == [1, 2]
