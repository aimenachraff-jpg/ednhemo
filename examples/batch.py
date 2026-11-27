"""Batch analysis of a folder of NIfTI scans.

    python examples/batch.py path/to/scans/ --model variante --out results/

Writes one CSV line per scan and keeps every per-scan output folder.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ednhemo import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="folder containing .nii / .nii.gz scans")
    parser.add_argument("--model", default="variante", choices=["socle", "variante"])
    parser.add_argument("--out", default="batch_results")
    args = parser.parse_args()

    scans = sorted(
        path
        for path in Path(args.folder).iterdir()
        if path.name.lower().endswith((".nii", ".nii.gz"))
    )
    if not scans:
        raise SystemExit(f"No NIfTI file found in {args.folder}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, scan in enumerate(scans, start=1):
        print(f"[{index}/{len(scans)}] {scan.name}")
        result = analyze(scan, model=args.model)
        result.save(out / scan.name.replace(".nii.gz", "").replace(".nii", ""))
        rows.append(
            {
                "scan": scan.name,
                "n_lesions": result.n_lesions,
                "n_under_5ml": result.n_small_lesions,
                "total_volume_ml": round(result.total_volume_ml, 3),
                "positive_voxels": result.positive_voxels,
            }
        )

    summary = out / "batch_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
