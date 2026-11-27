"""
utils.py - fonctions partagées du mini-projet M2 « Les petites hémorragies
intracrâniennes au scanner » (équipe 36, lot G).

Ce fichier est écrit une seule fois et importé par les six notebooks.
Il ne contient que des fonctions génériques : chargement de BHSD, découpage
officiel, fenêtrage CT, extraction des lésions (composantes connexes 3D),
tableaux de synthèse et figures.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

# ---------------------------------------------------------------------------
# Constantes communes à toutes les équipes
# ---------------------------------------------------------------------------

# Sous-types d'hémorragie codés dans les masques BHSD
SOUS_TYPES = {
    1: "épidural",
    2: "intraparenchymateux",
    3: "intraventriculaire",
    4: "sous-arachnoïdien",
    5: "sous-dural",
}
SOUS_TYPES_INV = {v: k for k, v in SOUS_TYPES.items()}

# Classes de taille imposées par le script d'évaluation commun
CLASSES = ["< 0,1 mL", "0,1 à 1 mL", "1 à 5 mL", "> 5 mL"]

# Une lésion (référence) ou une fausse alarme (prédiction) doit faire au moins
# 0,01 mL, soit environ 8 voxels : en dessous, c'est un reste d'annotation.
VOLUME_MIN_ML = 0.01

# Deux voxels qui se touchent par une face, une arête ou un coin sont connectés
CONNEXITE_3D = np.ones((3, 3, 3), dtype=bool)

# Fenêtre de densité du cerveau imposée : [-40, 120] UH, puis (valeur - 40) / 80
CT_MIN_HU = -40.0
CT_MAX_HU = 120.0


# ---------------------------------------------------------------------------
# Localisation des fichiers (Kaggle ou poste local)
# ---------------------------------------------------------------------------

def _has_nifti(p: Path) -> bool:
    """Vérifie que le dossier contient au moins un fichier NIfTI (.nii ou .nii.gz)."""
    return p.is_dir() and any(p.glob("*.nii*"))


def find_bhsd_root(start: str | Path | None = None) -> Path:
    """Trouve le dossier BHSD qui contient ``images/`` et ``ground truths/``.

    Cherche dans cet ordre : variable d'environnement BHSD_ROOT, chemin fourni,
    /kaggle/input/bhsd-label-192/label_192 (structure réelle du dataset Kaggle),
    /kaggle/input/bhsd-label-192, /kaggle/input, dossier courant,
    dossier de utils.py. Compatible avec un zip Kaggle décompressé sous différents
    noms (label_192, BHSD, ...).
    """
    env = os.environ.get("BHSD_ROOT")
    if env:
        p = Path(env)
        if _has_nifti(p / "images") and (p / "ground truths").is_dir():
            return p

    starts: list[Path] = []
    if start is not None:
        starts.append(Path(start))
    # Mount points Kaggle explicites — la structure réelle est label_192/ à la racine
    starts += [
        Path("/kaggle/input/bhsd-label-192/label_192"),
        Path("/kaggle/input/bhsd-label-192"),
        Path("/kaggle/input/datasets/aymencharafedn/bhsd-label-192/label_192"),
        Path("/kaggle/input/datasets/aymencharafedn/bhsd-label-192"),
        Path("/kaggle/input/aymencharafedn/bhsd-label-192/label_192"),
        Path("/kaggle/input/aymencharafedn/bhsd-label-192"),
        Path("/kaggle/input"),
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]

    deja_vus: set[str] = set()
    for s in starts:
        if not s.exists() or str(s) in deja_vus:
            continue
        deja_vus.add(str(s))
        candidats = [s, s / "label_192"]
        for c in candidats:
            if _has_nifti(c / "images") and (c / "ground truths").is_dir():
                return c
        # Dernier recours : chercher récursivement un dossier avec images + ground truths
        for images in s.rglob("images"):
            if images.is_dir() and _has_nifti(images) and (images.parent / "ground truths").is_dir():
                return images.parent

    raise FileNotFoundError(
        "Dossier BHSD introuvable. Indiquez-le avec la variable d'environnement "
        "BHSD_ROOT ou l'argument start=... (il doit contenir 'images' et "
        "'ground truths' avec des fichiers NIfTI)."
    )


def find_validation_json(start: str | Path | None = None) -> Path:
    """Trouve validation_bhsd.json (liste fixe des 16 scanners de validation)."""
    env = os.environ.get("BHSD_VALIDATION_JSON")
    if env and Path(env).is_file():
        return Path(env)

    starts: list[Path] = []
    if start is not None:
        starts.append(Path(start))
    starts += [
        Path("/kaggle/input/m2-hemorragies-equipe-36"),
        Path("/kaggle/input/m2-hemorragies-equipe-36/notebooks"),
        Path("/kaggle/input/datasets/aymencharafedn/m2-hemorragies-equipe-36"),
        Path("/kaggle/input/datasets/aymencharafedn/m2-hemorragies-equipe-36/notebooks"),
        Path("/kaggle/input/aymencharafedn/m2-hemorragies-equipe-36"),
        Path("/kaggle/input/aymencharafedn/m2-hemorragies-equipe-36/notebooks"),
        Path("/kaggle/working"),
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]

    for s in starts:
        for c in (s / "validation_bhsd.json", s / "notebooks" / "validation_bhsd.json"):
            if c.is_file():
                return c

    # Dernier recours sur Kaggle : chercher récursivement
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for c in kaggle_input.rglob("validation_bhsd.json"):
            if c.is_file():
                return c

    raise FileNotFoundError(
        "validation_bhsd.json introuvable : joignez le dataset de l'équipe "
        "ou copiez le fichier à côté de utils.py."
    )


def list_case_ids(root: str | Path) -> list[str]:
    """Liste triée des identifiants de scanners (noms de fichiers sans extension .nii ou .nii.gz)."""
    root = Path(root)
    ids = sorted(
        p.name.replace(".nii.gz", "").replace(".nii", "")
        for p in (root / "images").glob("*.nii*")
    )
    if not ids:
        raise FileNotFoundError(f"Aucune image NIfTI dans {root / 'images'}")
    return ids


def load_splits(
    root: str | Path | None = None,
    validation_json: str | Path | None = None,
) -> dict[str, list[str]]:
    """Construit le découpage officiel : 80 entraînement, 16 validation, 96 test.

    BHSD est officiellement découpé 96/96. La liste fixe fournie par
    l'enseignante donne 16 scanners de validation pris dans les 96
    d'entraînement. Les 96 scanners restants forment le test.
    """
    root = find_bhsd_root(root)
    ids = set(list_case_ids(root))

    vp = Path(validation_json) if validation_json else find_validation_json()
    with open(vp, encoding="utf-8") as f:
        d = json.load(f)
    train = list(d["train"])
    val = list(d["val"])

    inconnus = (set(train) | set(val)) - ids
    if inconnus:
        raise ValueError(f"Identifiants du JSON absents de BHSD : {sorted(inconnus)[:5]} ...")
    if set(train) & set(val):
        raise ValueError("Un scanner est à la fois dans train et dans val.")

    test = sorted(ids - set(train) - set(val))
    splits = {"train": sorted(train), "val": sorted(val), "test": test}

    attendus = {"train": 80, "val": 16, "test": 96}
    for nom, n in attendus.items():
        if len(splits[nom]) != n:
            raise ValueError(f"Découpage {nom} : {len(splits[nom])} scanners au lieu de {n}.")
    return splits


# ---------------------------------------------------------------------------
# Chargement et prétraitement
# ---------------------------------------------------------------------------

def _find_case_file(d: Path, case_id: str) -> Path:
    """Trouve le fichier NIfTI d'un scanner (.nii.gz ou .nii)."""
    for ext in (".nii.gz", ".nii"):
        p = d / f"{case_id}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Aucun fichier NIfTI pour {case_id} dans {d}")


def load_case(root: str | Path, case_id: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Charge l'image CT, le masque de référence et l'espacement (mm) d'un scanner."""
    root = Path(root)
    img = nib.load(_find_case_file(root / "images", case_id))
    gt = nib.load(_find_case_file(root / "ground truths", case_id))
    image = np.asarray(img.dataobj, dtype=np.float32)
    mask = np.asarray(gt.dataobj).astype(np.int16)
    if image.shape != mask.shape:
        raise ValueError(f"Tailles différentes pour {case_id} : {image.shape} / {mask.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return image, mask, zooms


def volume_voxel_ml(zooms) -> float:
    """Volume d'un voxel en mL (1 mL = 1000 mm³)."""
    return float(np.prod(zooms)) / 1000.0


def window_ct(image: np.ndarray) -> np.ndarray:
    """Fenêtre cérébrale imposée : clip [-40, 120] UH puis (valeur - 40) / 80."""
    image = np.clip(np.asarray(image, dtype=np.float32), CT_MIN_HU, CT_MAX_HU)
    return (image - 40.0) / 80.0


def classe_de_taille(volume_ml: float) -> str:
    """Classe de taille imposée par l'évaluation commune."""
    if volume_ml < 0.1:
        return CLASSES[0]
    if volume_ml < 1.0:
        return CLASSES[1]
    if volume_ml < 5.0:
        return CLASSES[2]
    return CLASSES[3]


# ---------------------------------------------------------------------------
# Extraction des lésions (phase commune)
# ---------------------------------------------------------------------------

COLONNES_LESIONS = [
    "scanner", "split", "lesion", "n_voxels", "volume_ml", "classe_taille",
    "sous_type", "sous_type_id", "z_min", "z_max", "z_max_lesion",
    "centroid_z", "centroid_y", "centroid_x",
]


def extract_lesions(
    mask: np.ndarray,
    zooms,
    scanner: str,
    split: str = "",
    min_volume_ml: float = VOLUME_MIN_ML,
) -> pd.DataFrame:
    """Extrait chaque hémorragie comme composante connexe 3D.

    Retourne une ligne par lésion d'au moins ``min_volume_ml`` avec son volume
    (mL), sa classe de taille, son sous-type majoritaire et sa position
    (bornes en z, coupe la plus touchée, centroïde).
    """
    mask = np.asarray(mask).astype(np.int16)
    binaire = mask > 0
    lab, n = ndimage.label(binaire, structure=CONNEXITE_3D)
    voxel_ml = volume_voxel_ml(zooms)
    tranches = ndimage.find_objects(lab)

    lignes = []
    for k, sl in enumerate(tranches, start=1):
        if sl is None:
            continue
        comp = lab[sl] == k
        n_vox = int(comp.sum())
        volume = n_vox * voxel_ml
        if volume < min_volume_ml:
            continue

        valeurs = mask[sl][comp]
        vals, comptes = np.unique(valeurs, return_counts=True)
        sous_type_id = int(vals[np.argmax(comptes)])
        sous_type = SOUS_TYPES.get(sous_type_id, "inconnu")

        # Les volumes BHSD sont stockés en (y, x, z) : l'axe 2 est l'axe des coupes.
        # Coupe axiale où la lésion est la plus visible (utile pour les figures)
        par_coupe = comp.sum(axis=(0, 1))
        z_local = int(np.argmax(par_coupe))
        z_max_lesion = int(sl[2].start + z_local)

        yy, xx, zz = np.nonzero(comp)
        cy = float(sl[0].start + yy.mean())
        cx = float(sl[1].start + xx.mean())
        cz = float(sl[2].start + zz.mean())

        lignes.append({
            "scanner": scanner,
            "split": split,
            "n_voxels": n_vox,
            "volume_ml": round(float(volume), 6),
            "classe_taille": classe_de_taille(volume),
            "sous_type": sous_type,
            "sous_type_id": sous_type_id,
            "z_min": int(sl[2].start),
            "z_max": int(sl[2].stop - 1),
            "z_max_lesion": z_max_lesion,
            "centroid_z": round(cz, 2),
            "centroid_y": round(cy, 2),
            "centroid_x": round(cx, 2),
        })

    df = pd.DataFrame(lignes, columns=COLONNES_LESIONS[3:])
    df.insert(0, "split", split)
    df.insert(0, "scanner", scanner)
    df.insert(2, "lesion", np.arange(1, len(df) + 1))
    return df


def lesions_du_dataset(
    root: str | Path,
    splits: dict[str, list[str]],
    verbose: bool = True,
) -> pd.DataFrame:
    """Extrait les lésions des 192 scanners et concatène les résultats."""
    root = Path(root)
    split_de = {cid: nom for nom, ids in splits.items() for cid in ids}
    morceaux = []
    ids = splits["train"] + splits["val"] + splits["test"]

    iterable = ids
    if verbose:
        from tqdm.auto import tqdm
        iterable = tqdm(ids, desc="Extraction des lésions")

    for cid in iterable:
        gt = nib.load(_find_case_file(root / "ground truths", cid))
        mask = np.asarray(gt.dataobj).astype(np.int16)
        morceaux.append(extract_lesions(mask, gt.header.get_zooms()[:3], cid, split_de[cid]))

    lesions = pd.concat(morceaux, ignore_index=True)
    lesions["split"] = pd.Categorical(
        lesions["split"], categories=["train", "val", "test"], ordered=True
    )
    return lesions


# ---------------------------------------------------------------------------
# Tableaux de synthèse
# ---------------------------------------------------------------------------

def tableau_scanners(lesions: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    """Une ligne par scanner, y compris ceux sans lésion (compteurs à zéro)."""
    split_de = {cid: nom for nom, ids in splits.items() for cid in ids}
    ordre = splits["train"] + splits["val"] + splits["test"]
    lignes = []
    for cid in ordre:
        d = lesions[lesions["scanner"] == cid]
        lignes.append({
            "scanner": cid,
            "split": split_de[cid],
            "n_lesions": len(d),
            "volume_total_ml": round(float(d["volume_ml"].sum()), 4),
            "n_moins_0_1": int((d["classe_taille"] == CLASSES[0]).sum()),
            "n_0_1_a_1": int((d["classe_taille"] == CLASSES[1]).sum()),
            "n_1_a_5": int((d["classe_taille"] == CLASSES[2]).sum()),
            "n_plus_5": int((d["classe_taille"] == CLASSES[3]).sum()),
        })
    return pd.DataFrame(lignes)


def tableau_par_classe(lesions: pd.DataFrame) -> pd.DataFrame:
    """Nombre de lésions, volume et part de chaque classe de taille."""
    total = len(lesions)
    volume_total = float(lesions["volume_ml"].sum())
    lignes = []
    for c in CLASSES:
        d = lesions[lesions["classe_taille"] == c]
        lignes.append({
            "classe_taille": c,
            "n_lesions": len(d),
            "pct_lesions": round(100 * len(d) / total, 2) if total else 0.0,
            "volume_total_ml": round(float(d["volume_ml"].sum()), 3),
            "pct_volume": round(100 * float(d["volume_ml"].sum()) / volume_total, 2) if volume_total else 0.0,
            "volume_median_ml": round(float(d["volume_ml"].median()), 4) if len(d) else None,
        })
    return pd.DataFrame(lignes)


def tableau_par_sous_type(lesions: pd.DataFrame) -> pd.DataFrame:
    """Nombre de lésions, volume et part de chaque sous-type."""
    total = len(lesions)
    volume_total = float(lesions["volume_ml"].sum())
    lignes = []
    for st in SOUS_TYPES.values():
        d = lesions[lesions["sous_type"] == st]
        lignes.append({
            "sous_type": st,
            "n_lesions": len(d),
            "pct_lesions": round(100 * len(d) / total, 2) if total else 0.0,
            "volume_total_ml": round(float(d["volume_ml"].sum()), 3),
            "pct_volume": round(100 * float(d["volume_ml"].sum()) / volume_total, 2) if volume_total else 0.0,
            "volume_median_ml": round(float(d["volume_ml"].median()), 4) if len(d) else None,
        })
    return pd.DataFrame(lignes)


def tableau_croise(lesions: pd.DataFrame) -> pd.DataFrame:
    """Tableau croisé nombre de lésions : classes de taille x sous-types."""
    return pd.crosstab(lesions["sous_type"], lesions["classe_taille"]).reindex(
        index=list(SOUS_TYPES.values()), columns=CLASSES, fill_value=0
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_distribution_tailles(lesions: pd.DataFrame, sortie: str | Path | None = None):
    """Histogramme log des volumes + barres par classe de taille."""
    volumes = lesions["volume_ml"].to_numpy()
    if len(volumes) == 0:
        raise ValueError("Aucune lésion à tracer.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    vmin = max(VOLUME_MIN_ML, float(volumes.min()))
    bins = np.logspace(np.log10(vmin), np.log10(max(float(volumes.max()), 10.0)), 45)
    ax.hist(volumes, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ymax = ax.get_ylim()[1]
    for seuil, nom in [(0.1, "0,1 mL"), (1, "1 mL"), (5, "5 mL")]:
        ax.axvline(seuil, color="#C44E52", linestyle="--", linewidth=1.2)
        ax.text(seuil * 0.92, ymax * 0.96, nom, color="#C44E52", rotation=90,
                fontsize=8, va="top", ha="right")
    ax.set_xlabel("Volume de la lésion (mL, échelle log)")
    ax.set_ylabel("Nombre de lésions")
    ax.set_title(f"Distribution des tailles ({len(volumes)} lésions, "
                 f"{lesions['scanner'].nunique()} scanners)")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    ordre = CLASSES
    comptes = [int((lesions["classe_taille"] == c).sum()) for c in ordre]
    couleurs = ["#55A868", "#4C72B0", "#DD8452", "#C44E52"]
    barres = ax.bar(ordre, comptes, color=couleurs)
    for b, n in zip(barres, comptes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{n}\n({100 * n / len(volumes):.1f} %)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Nombre de lésions")
    ax.set_title("Lésions par classe de taille")
    ax.set_ylim(0, max(comptes) * 1.22 if comptes else 1)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    if sortie is not None:
        sortie = Path(sortie)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sortie, dpi=160, bbox_inches="tight")
    return fig


def plot_exemples_lesions(
    root: str | Path,
    lesions: pd.DataFrame,
    sortie: str | Path | None = None,
    n: int = 4,
    volume_max: float = 1.0,
):
    """Affiche n petites lésions : coupe CT fenêtrée + contour du masque de référence."""
    root = Path(root)
    candidates = lesions[lesions["volume_ml"] <= volume_max].sort_values("volume_ml")
    if len(candidates) < n:
        candidates = lesions.sort_values("volume_ml")
    if len(candidates) > n:
        # Prendre des lésions réparties dans la plage des petites tailles
        idx = np.linspace(0, len(candidates) - 1, n).round().astype(int)
        candidates = candidates.iloc[idx]
    candidates = candidates.head(n)

    fig, axes = plt.subplots(1, len(candidates), figsize=(4 * len(candidates), 4.2))
    if len(candidates) == 1:
        axes = [axes]

    for ax, (_, l) in zip(axes, candidates.iterrows()):
        image, mask, _ = load_case(root, l["scanner"])
        z = int(l["z_max_lesion"])
        coupe = window_ct(image[:, :, z])
        ax.imshow(np.rot90(coupe), cmap="gray", vmin=-1, vmax=1)
        masque_coupe = np.rot90(mask[:, :, z] > 0)
        ax.imshow(np.ma.masked_where(~masque_coupe, masque_coupe), cmap="autumn", alpha=0.45)
        ax.set_title(f"{l['sous_type']}\n{l['volume_ml']:.3f} mL - coupe {z}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Exemples de petites hémorragies (rouge : masque de référence)", fontsize=11)
    fig.tight_layout()
    if sortie is not None:
        sortie = Path(sortie)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sortie, dpi=160, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fixe la graine Python / NumPy (PyTorch est fixé dans les notebooks)."""
    random.seed(seed)
    np.random.seed(seed)
