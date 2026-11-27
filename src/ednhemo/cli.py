"""Command-line interface: ``ednhemo scan.nii.gz``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .inference import analyze
from .weights import MODELS, list_models

__all__ = ["main", "format_report"]


def format_report(result: object) -> str:
    """Plain-text report printed by the CLI."""
    from .inference import DetectionResult

    assert isinstance(result, DetectionResult)
    info = MODELS[result.model]
    lines = [
        f"ednhemo {__version__}",
        f"Input      : {result.source}",
        f"Model      : {result.model} - {info['label']}",
        f"             Dice {info['dice']:.3f} | sensitivity {info['sensitivity']:.1f}% | "
        f"false alarms/scan {info['false_alarms_per_scan']:.2f}",
        f"Volume     : {result.n_slices} slice(s), "
        f"{result.spacing[0]:.3f} x {result.spacing[1]:.3f} x {result.spacing[2]:.1f} mm",
        f"Mask       : {result.positive_voxels:,} positive voxel(s) "
        f"(threshold {result.threshold:.2f})",
        f"Detected   : {result.n_lesions} hemorrhage(s), "
        f"{result.n_small_lesions} under 5 mL, {result.total_volume_ml:.3f} mL total",
    ]
    if not result.lesions:
        lines.append("")
        lines.append("No hemorrhage detected at this threshold.")
        return "\n".join(lines)

    header = f"{'#':>3}  {'slice':>5}  {'volume_ml':>10}  {'size class':<11}  {'voxels':>7}  {'confidence':>10}"
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for lesion in result.lesions:
        lines.append(
            f"{lesion.index:>3}  {lesion.slice:>5}  {lesion.volume_ml:>10.3f}  "
            f"{lesion.size_class:<11}  {lesion.voxels:>7,}  {lesion.confidence:>10.3f}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ednhemo",
        description=(
            "Intracranial hemorrhage detection on CT with a small-lesion-aware 2D U-Net. "
            "Research prototype - not a medical device."
        ),
        epilog="Example: ednhemo scan.nii.gz --model variante --out results/",
    )
    parser.add_argument("path", nargs="?", help="NIfTI scan (.nii/.nii.gz) or 2D image (.png/.jpg)")
    parser.add_argument(
        "-m",
        "--model",
        default="variante",
        choices=["socle", "variante", "base", "small", "lot-g"],
        help="checkpoint to use (default: variante)",
    )
    parser.add_argument("-o", "--out", help="output directory (default: ednhemo_output/<name>_<model>)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="probability threshold (default: 0.5)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=8, help="slices per forward pass (default: 8)")
    parser.add_argument("--min-volume", type=float, default=0.01, help="ignore lesions below this volume in mL")
    parser.add_argument("--no-figure", action="store_true", help="skip the PNG overlay")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--list-models", action="store_true", help="list the shipped checkpoints and exit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print(f"ednhemo {__version__} - available checkpoints")
        print()
        for info in list_models():
            print(f"  {info['name']:<9} {info['label']}")
            print(
                f"            Dice {info['dice']:.3f} | sensitivity {info['sensitivity']:.1f}% | "
                f"false alarms/scan {info['false_alarms_per_scan']:.2f} | "
                f"validation Dice {info['validation_dice']:.4f} (epoch {info['best_epoch']})"
            )
        return 0

    if not args.path:
        parser.print_help()
        return 0

    try:
        result = analyze(
            args.path,
            model=args.model,
            threshold=args.threshold,
            device=args.device,
            batch_size=args.batch_size,
            min_volume_ml=args.min_volume,
            figure=not args.no_figure,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ednhemo: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
    else:
        print(format_report(result))

    if args.out:
        out = Path(args.out)
    else:
        stem = Path(args.path).stem.replace(".nii", "")
        out = Path("ednhemo_output") / f"{stem}_{result.model}"
    written = result.save(out)
    print()
    print(f"Outputs written to {out.resolve()}")
    for path in written:
        print(f"  - {path.name}")
    return 0
