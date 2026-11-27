"""Model architecture, checkpoint loading and shape checks."""

import pytest
import torch

from ednhemo import UNet2D, count_parameters, load_model


def test_parameter_count_matches_published_checkpoints():
    assert count_parameters(UNet2D()) == 7_765_409


def test_forward_preserves_spatial_shape():
    model = UNet2D()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 64, 64))
    assert output.shape == (2, 1, 64, 64)


def test_forward_handles_odd_sizes():
    model = UNet2D()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 100, 92))
    assert output.shape == (1, 1, 100, 92)


@pytest.mark.parametrize("name", ["socle", "variante"])
def test_checkpoints_load(name):
    model = load_model(name, device="cpu")
    assert not model.training
    assert count_parameters(model) == 7_765_409


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("does-not-exist", device="cpu")


def test_aliases_resolve():
    assert load_model("base", device="cpu") is load_model("socle", device="cpu")
    assert load_model("small", device="cpu") is load_model("variante", device="cpu")
