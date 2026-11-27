"""Minimal Python API example.

Run with a NIfTI scan or a 2D image:

    python examples/quickstart.py path/to/scan.nii.gz
"""

from __future__ import annotations

import sys
from pathlib import Path

from ednhemo import analyze


def main(path: str, model: str = "variante") -> None:
    result = analyze(path, model=model)

    print(f"{Path(path).name} - model {result.model}")
    print(f"{result.n_lesions} hemorrhage(s), {result.n_small_lesions} under 5 mL")
    print()

    for lesion in result.lesions:
        print(
            f"  slice {lesion.slice:>3} | {lesion.volume_ml:7.3f} mL | "
            f"{lesion.size_class:<10} | confidence {lesion.confidence:.3f}"
        )

    out = Path("ednhemo_output") / f"{Path(path).stem}_{result.model}"
    for written in result.save(out):
        print(f"  wrote {written}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python examples/quickstart.py SCAN [socle|variante]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "variante")
