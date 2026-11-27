# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-27

### Fixed

- Absolute links in the README so the PyPI project page resolves the license,
  citation and research links instead of pointing inside the PyPI domain.

## [0.1.0] - 2026-11-27

### Added

- First public release of `ednhemo`.
- `socle` (reference) and `variante` (lot G, small-lesion oversampling)
  checkpoints bundled in the wheel — no download required.
- `analyze()` Python API: NIfTI scan or 2D image in, lesions and mask out.
- `ednhemo` CLI with `--model`, `--threshold`, `--device`, `--json` and
  `--list-models`.
- Lesion-by-lesion report: volume in mL, size class, slice, voxel count and
  mean confidence.
- Outputs: `mask.nii.gz`, `lesions.csv`, `summary.json`, `overlay.png`.
- Exact reproduction of the training preprocessing: `[-40, 120]` HU window,
  `(v - 40) / 80` normalization, 512 × 512 slices, 0.5 threshold.
- Test suite (27 tests) and GitHub Actions CI.
