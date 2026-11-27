"""ednhemo - intracranial hemorrhage detection on CT.

Small-hemorrhage-aware 2D U-Net trained on the BHSD dataset (192 CT scans).
Two checkpoints are shipped: the reference model (``socle``) and the lot G
model (``variante``) trained with small hemorrhages guaranteed in every batch.

Example
-------
>>> from ednhemo import analyze
>>> result = analyze("scan.nii.gz", model="variante")
>>> result.n_lesions, result.n_small_lesions
(5, 4)
>>> result.save("out/")
[...]
"""

from ._version import __version__
from .inference import DetectionResult, analyze, predict_probabilities
from .lesions import Lesion, find_lesions, size_class
from .model import UNet2D, build_unet, count_parameters
from .preprocessing import (
    BHSD_SPACING,
    CT_MAX_HU,
    CT_MIN_HU,
    IMAGE_SIZE,
    Volume,
    load_volume,
    normalize_hu,
    normalize_windowed,
    resize_slice,
)
from .weights import MODELS, load_model, resolve_model_name

__all__ = [
    "__version__",
    "DetectionResult",
    "Lesion",
    "MODELS",
    "UNet2D",
    "Volume",
    "analyze",
    "build_unet",
    "count_parameters",
    "find_lesions",
    "load_model",
    "load_volume",
    "normalize_hu",
    "normalize_windowed",
    "predict_probabilities",
    "resize_slice",
    "resolve_model_name",
    "size_class",
    "BHSD_SPACING",
    "CT_MAX_HU",
    "CT_MIN_HU",
    "IMAGE_SIZE",
]
