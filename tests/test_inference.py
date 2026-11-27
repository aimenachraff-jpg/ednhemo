"""End-to-end inference on a tiny synthetic NIfTI."""

import json

import nibabel as nib
import numpy as np
import pytest

from ednhemo import analyze


@pytest.fixture(scope="module")
def tiny_scan(tmp_path_factory):
    path = tmp_path_factory.mktemp("scan") / "tiny.nii.gz"
    data = np.full((32, 32, 3), -1000.0, dtype=np.float32)
    data[8:24, 8:24, :] = 40.0
    data[12:16, 12:16, 1] = 75.0  # hyperdense blob
    affine = np.diag([0.5, 0.5, 5.0, 1.0])
    image = nib.Nifti1Image(data, affine)
    nib.save(image, str(path))
    return path


def test_analyze_returns_a_mask_at_input_resolution(tiny_scan):
    result = analyze(tiny_scan, model="socle", device="cpu", figure=False)

    assert result.mask.shape == (32, 32, 3)
    assert result.mask.dtype == bool
    assert result.model == "socle"
    assert result.is_nifti
    assert result.n_slices == 3


def test_analyze_can_keep_probabilities(tiny_scan):
    result = analyze(tiny_scan, model="socle", device="cpu", figure=False, keep_probabilities=True)
    assert result.probabilities is not None
    assert result.probabilities.shape == (32, 32, 3)
    assert float(result.probabilities.min()) >= 0.0
    assert float(result.probabilities.max()) <= 1.0


def test_save_writes_every_output(tiny_scan, tmp_path):
    result = analyze(tiny_scan, model="socle", device="cpu", figure=True)
    written = result.save(tmp_path / "out")

    names = {path.name for path in written}
    assert names == {"mask.nii.gz", "lesions.csv", "summary.json", "overlay.png"}
    assert all(path.is_file() for path in written)

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_lesions"] == len(result.lesions)
    assert summary["spacing_mm"] == [0.5, 0.5, 5.0]


def test_invalid_threshold_is_rejected(tiny_scan):
    with pytest.raises(ValueError, match="threshold"):
        analyze(tiny_scan, threshold=1.5, figure=False)
