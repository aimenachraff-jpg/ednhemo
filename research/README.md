# Research — the academic project behind `ednhemo`

This folder keeps the material produced for the M2 mini-project
**“Les petites hémorragies intracrâniennes au scanner”** (team 36, lot G).

The Python package at the root is the reusable part; everything here is the
evidence: notebooks, official results, figures and reports.

## What the project asked

Train a 2D U-Net on the public **BHSD** dataset (192 annotated CT scans) with an
imposed protocol (Dice + cross-entropy, Adam 1e-4, batch 8, 50 epochs,
512 × 512, window `[-40, 120]` HU, horizontal flip + ±10° rotation only,
threshold 0.5, seed 36, official 80/16/96 split).

Team 36 drew **lot G**: in every batch of 8 slices, at least 4 must contain a
hemorrhage **under 5 mL**. The question: does that improve small-lesion
detection, and at what cost in false alarms?

## Contents

| Folder | Content |
|---|---|
| `notebooks/` | The six required notebooks (`01`–`06`), the inference notebook (`07`) and `utils.py` |
| `results/` | Official per-lesion, per-scan, per-class CSVs and `comparaison.json` |
| `reports/` | One-page report, final report (5 pages), state-of-the-art matrix, A4 summary |
| `../docs/images/` | Figures used in the README and the reports |

## Official results

96 test scans, 864 lesions, `evaluation_commune.py` used **without modification**
(SHA-256 verified).

| Model | Dice | Sensitivity | False alarms / scan |
|---|---|---|---|
| `socle` (reference) | 0.510 | 41.9 % | 3.43 |
| `variante` (lot G) | **0.540** | **44.1 %** | 3.78 |

58 lesions missed by the reference are recovered by the lot G model, 39 are
lost (net +19); **52 of the 58 recovered lesions are under 1 mL**.

## Reproducing

The notebooks are published on Kaggle and run on the same datasets:

- kernels: `aymencharafedn/01-exploration-equipe-36` … `07-detection-equipe-36`
- datasets: `bhsd-label-192`, `m2-modeles-equipe-36`,
  `m2-predictions-socle-equipe-36`, `m2-predictions-variante-equipe-36`

Locally, `pip install ednhemo` reproduces the inference path exactly: on the
test scan `ID_0237f3c9_ID_40015688b9`, the package mask is identical
(Dice 1.0, 10,407 voxels, 5 lesions) to the project's published prediction.

## Team

- Chekalil Aimene Charef Eddine
- Ouadah Mustapha
