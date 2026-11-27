"""Input loading and normalization."""

import numpy as np
from PIL import Image

from ednhemo import BHSD_SPACING, load_volume, normalize_hu, normalize_windowed, resize_slice


def test_normalize_hu_maps_the_window_to_minus_one_one():
    values = np.array([-1000.0, -40.0, 40.0, 120.0, 3000.0], dtype=np.float32)
    normalized = normalize_hu(values)
    assert np.allclose(normalized, [-1.0, -1.0, 0.0, 1.0, 1.0])


def test_normalize_windowed_maps_8bit_to_minus_one_one():
    assert np.allclose(normalize_windowed(np.array([0.0, 127.5, 255.0])), [-1.0, 0.0, 1.0])


def test_load_2d_png(tmp_path):
    path = tmp_path / "slice.png"
    Image.fromarray(np.full((32, 48), 128, dtype=np.uint8)).save(path)

    volume = load_volume(path)

    assert volume.data.shape == (32, 48, 1)
    assert volume.spacing == BHSD_SPACING
    assert volume.affine is None
    assert not volume.is_nifti
    assert -1.0 <= float(volume.data.min()) and float(volume.data.max()) <= 1.0


def test_load_numpy_array_is_treated_as_hu():
    volume = load_volume(np.full((16, 16), 40.0, dtype=np.float32))
    assert volume.data.shape == (16, 16, 1)
    assert np.allclose(volume.data, 0.0)


def test_resize_slice_outputs_512():
    resized = resize_slice(np.zeros((100, 140), dtype=np.float32))
    assert resized.shape == (512, 512)


def test_resize_slice_is_a_noop_on_512():
    array = np.random.default_rng(36).random((512, 512), dtype=np.float32)
    assert np.allclose(resize_slice(array), array)
