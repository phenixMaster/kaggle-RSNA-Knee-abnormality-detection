"""
Génération du fichier submission.csv pour la compétition
RSNA Knee Abnormality Detection.

Le fichier submission.csv contient les probabilités prédites par le modèle
pour chacun des 12 labels de la compétition, une ligne par StudyInstanceUID.
Il est soumis directement au scorer Kaggle (métrique : moyenne des AUC ROC).

Deux modes d'utilisation :
  1. Import dans un notebook Kaggle (recommandé) :
        from generate_submission import make_submission
        make_submission(study_ids, predictions, output="/kaggle/working/submission.csv")

  2. CLI (soumission de base avec score par défaut) :
        python generate_submission.py
        python generate_submission.py \\
            --test_series /kaggle/input/.../test_series.csv \\
            --output /kaggle/working/submission.csv \\
            --default_score 0.5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# --- Configuration ---
# 12 labels binaires de la compétition (ordre identique au sample_submission.csv)
TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Chemins Kaggle par défaut
KAGGLE_TEST_SERIES = Path(
    "/kaggle/input/competitions/rsna-knee-abnormality-detection/test_series.csv"
)
KAGGLE_OUTPUT_CSV = Path("/kaggle/working/submission.csv")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fonctions publiques (importables depuis un notebook)
# ---------------------------------------------------------------------------


def load_study_ids(test_series: Union[str, Path]) -> pd.Series:
    """
    @definition : Charge les StudyInstanceUID uniques depuis test_series.csv.
                  Plusieurs séries (Sagittal, Axial, Coronal) existent par étude ;
                  on déduplique pour obtenir une ligne par étude.
    @args/params : test_series (str | Path) — chemin vers test_series.csv sur Kaggle.
    @return : pd.Series de StudyInstanceUID uniques, triés alphabétiquement.
    """
    test_series = Path(test_series)
    if not test_series.exists():
        logger.error(f"Fichier introuvable : {test_series}")
        sys.exit(1)

    df = pd.read_csv(test_series, usecols=["StudyInstanceUID"])
    n_series = len(df)

    # Déduplication — une ligne de soumission par étude
    uids = df["StudyInstanceUID"].drop_duplicates().sort_values().reset_index(drop=True)
    logger.info(f"test_series.csv : {n_series} séries → {len(uids)} études uniques")
    return uids


def make_submission(
    study_ids: Union[pd.Series, list],
    predictions: Union[np.ndarray, pd.DataFrame, None] = None,
    default_score: float = 0.5,
    output: Union[str, Path] = KAGGLE_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    @definition : Construit et sauvegarde submission.csv à partir des prédictions
                  du modèle. Chaque colonne correspond à un des 12 labels de la
                  compétition ; les valeurs sont des probabilités (0.0–1.0).
                  Le scorer Kaggle calcule la moyenne des AUC ROC sur les 12 labels.
    @args/params :
        study_ids     (pd.Series | list)             — StudyInstanceUID uniques
                                                       (même ordre que predictions).
        predictions   (np.ndarray | pd.DataFrame | None)
                      — Prédictions du modèle :
                        * np.ndarray  : shape (N, 12), colonnes dans l'ordre de
                                        TARGET_COLUMNS.
                        * pd.DataFrame: colonnes nommées selon TARGET_COLUMNS.
                        * None        : remplace par default_score (baseline).
        default_score (float)                        — valeur utilisée si
                                                       predictions is None (défaut 0.5).
        output        (str | Path)                   — chemin de sortie du CSV.
    @return : pd.DataFrame du fichier de soumission (aussi sauvegardé sur disque).
    """
    output = Path(output)
    study_ids = list(study_ids)
    n = len(study_ids)

    # --- Construction du DataFrame de prédictions ---
    if predictions is None:
        # Baseline : score uniforme pour tous les labels
        logger.warning(
            f"Aucune prédiction fournie — score par défaut {default_score} appliqué."
        )
        pred_df = pd.DataFrame(
            np.full((n, len(TARGET_COLUMNS)), default_score),
            columns=TARGET_COLUMNS,
        )

    elif isinstance(predictions, np.ndarray):
        # Array NumPy (N, 12) — colonnes dans l'ordre TARGET_COLUMNS
        if predictions.shape != (n, len(TARGET_COLUMNS)):
            raise ValueError(
                f"predictions.shape={predictions.shape} attendu ({n}, {len(TARGET_COLUMNS)})"
            )
        pred_df = pd.DataFrame(predictions, columns=TARGET_COLUMNS)

    elif isinstance(predictions, pd.DataFrame):
        # DataFrame avec colonnes nommées
        missing = set(TARGET_COLUMNS) - set(predictions.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans predictions : {missing}")
        pred_df = predictions[TARGET_COLUMNS].reset_index(drop=True)

    else:
        raise TypeError(
            f"Type non supporté pour predictions : {type(predictions)}. "
            "Utiliser np.ndarray, pd.DataFrame ou None."
        )

    # --- Assemblage final ---
    submission = pd.DataFrame({"StudyInstanceUID": study_ids})
    submission = pd.concat([submission, pred_df], axis=1)

    # Validation des bornes [0, 1]
    out_of_range = ((pred_df < 0) | (pred_df > 1)).any().any()
    if out_of_range:
        logger.warning("Certaines prédictions sont hors de [0, 1] — vérifier le modèle.")

    # Sauvegarde
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    logger.info(f"submission.csv sauvegardé : {output}")
    logger.info(f"  → {len(submission)} études × {len(TARGET_COLUMNS)} labels")
    logger.info(f"Aperçu :\n{submission.head(3).to_string(index=False)}")

    return submission


# ---------------------------------------------------------------------------
# CLI — utilisation directe (score par défaut, sans modèle)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """
    @definition : Parse les arguments de la ligne de commande.
    @args/params : Aucun (lit sys.argv).
    @return : Namespace avec test_series, output, default_score.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Génère submission.csv (baseline) pour RSNA Knee Abnormality Detection.\n"
            "Pour soumettre les prédictions d'un modèle, importer make_submission() "
            "directement dans le notebook Kaggle."
        )
    )
    parser.add_argument(
        "--test_series",
        type=Path,
        default=KAGGLE_TEST_SERIES,
        help=f"Chemin vers test_series.csv (défaut : {KAGGLE_TEST_SERIES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=KAGGLE_OUTPUT_CSV,
        help=f"Chemin de sortie (défaut : {KAGGLE_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--default_score",
        type=float,
        default=0.5,
        help="Score par défaut pour chaque label (défaut : 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    """
    @definition : Point d'entrée CLI — génère un fichier de soumission baseline
                  (sans prédictions de modèle) à partir de test_series.csv.
    @args/params : Aucun (utilise _parse_args).
    @return : None
    """
    args = _parse_args()
    logger.info("=== Génération du fichier de soumission (baseline) ===")

    # Chargement des StudyInstanceUID depuis test_series.csv
    uids = load_study_ids(args.test_series)

    # Construction et sauvegarde
    make_submission(
        study_ids=uids,
        predictions=None,
        default_score=args.default_score,
        output=args.output,
    )

    logger.info("=== Terminé ===")


if __name__ == "__main__":
    main()
