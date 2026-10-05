"""
Modèle baseline 2D pour RSNA Knee Abnormality Detection.

Simplifié depuis main.py :
  - Modèle 2D (1 coupe centrale par série, pas de stacking 2.5D)
  - Pas de fusion multimodale (pas de labels Gemma)
  - Pas d'ensemble (1 seul modèle, seed fixe)
  - Weighted BCE Loss calculée depuis train.csv
  - Génération automatique de submission.csv via generate_submission

Usage sur Kaggle :
    python baseline_submission.py
    python baseline_submission.py --epochs 5 --batch_size 8
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.utils.generate_submission import load_study_ids, make_submission
from src.utils.image_utils import load_dicom_image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# ---------------------------------------------------------------------------
# Dataset 2D simplifié (1 coupe centrale, pas de stacking)
# ---------------------------------------------------------------------------


class KneeDataset2D(Dataset):
    """
    Dataset 2D : charge une seule coupe centrale par série DICOM.
    Pas de stacking 2.5D, pas de labels textuels.
    """

    IMG_SIZE = 224

    def __init__(
        self,
        labels_csv: str,
        series_csv: str,
        images_dir: str,
        transform=None,
    ):
        """
        @definition : Charge et fusionne train.csv et train_series.csv.
                      Chaque ligne = une série (une coupe centrale sera chargée).
        @args/params :
            labels_csv  (str) — chemin vers train.csv.
            series_csv  (str) — chemin vers train_series.csv.
            images_dir  (str) — répertoire racine des fichiers DICOM.
            transform         — transformations PyTorch optionnelles.
        @return : Dataset PyTorch indexable.
        """
        labels_df = pd.read_csv(labels_csv)
        series_df = pd.read_csv(series_csv)

        # Jointure : une ligne par série, avec les labels de l'étude associée
        self.data = pd.merge(series_df, labels_df, on="StudyInstanceUID")
        self.images_dir = images_dir
        self.transform = transform

        logger.info(
            f"Dataset 2D initialisé : {len(self.data)} séries "
            f"({self.data['StudyInstanceUID'].nunique()} études)"
        )

    def __len__(self) -> int:
        """
        @definition : Nombre total de séries dans le dataset.
        @args/params : Aucun.
        @return : int — nombre de lignes.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        @definition : Charge la coupe centrale d'une série DICOM et retourne
                      l'image et les labels associés.
        @args/params : idx (int) — index de la série.
        @return : dict avec clés 'image' (Tensor 3×224×224) et
                  'labels' (Tensor 12,).
        """
        row = self.data.iloc[idx]
        study_id = row["StudyInstanceUID"]
        series_id = row["SeriesInstanceUID"]

        series_dir = os.path.join(self.images_dir, str(study_id), str(series_id))

        # Chargement de la coupe centrale
        image = self._load_central_slice(series_dir)

        if self.transform:
            image = self.transform(image)

        # Labels binaires — sans fusion multimodale
        labels = row[TARGET_COLUMNS].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=0.0)

        return {
            "image": image,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    def _load_central_slice(self, series_dir: str) -> torch.Tensor:
        """
        @definition : Charge la coupe DICOM centrale d'une série et la
                      normalise en Min-Max. Retourne un tenseur RGB (3, H, W).
        @args/params : series_dir (str) — répertoire de la série DICOM.
        @return : torch.Tensor de shape (3, 224, 224).
        """
        fallback = torch.zeros(3, self.IMG_SIZE, self.IMG_SIZE)

        if not os.path.exists(series_dir):
            return fallback

        files = sorted([f for f in os.listdir(series_dir) if f.endswith(".dcm")])
        if not files:
            return fallback

        # Coupe centrale
        central_idx = len(files) // 2
        dicom_path = os.path.join(series_dir, files[central_idx])
        img = load_dicom_image(dicom_path)

        if img is None:
            return fallback

        # Normalisation Min-Max → [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)

        # Resize → 224×224
        img_tensor = T.functional.resize(
            img_tensor,
            [self.IMG_SIZE, self.IMG_SIZE],
            interpolation=T.InterpolationMode.BICUBIC,
        )

        # Répliquer en 3 canaux (grayscale → RGB pour le backbone)
        img_tensor = img_tensor.repeat(3, 1, 1)

        return img_tensor


# ---------------------------------------------------------------------------
# Modèle 2D : EfficientNet-B0 + tête de classification
# ---------------------------------------------------------------------------


class KneeModel2D(nn.Module):
    """
    Modèle 2D léger : EfficientNet-B0 pré-entraîné avec tête multi-label.
    Entrée attendue : (B, 3, 224, 224).
    """

    def __init__(self, num_classes: int = 12):
        """
        @definition : Initialise EfficientNet-B0 et remplace la tête de
                      classification par une tête multi-label à num_classes sorties.
        @args/params : num_classes (int) — nombre de labels (défaut 12).
        @return : None
        """
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # Remplacement de la tête de classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @definition : Passe avant du modèle.
        @args/params : x (torch.Tensor) — batch d'images (B, 3, 224, 224).
        @return : torch.Tensor de logits (B, 12).
        """
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Calcul des poids de classe depuis train.csv
# ---------------------------------------------------------------------------


def compute_pos_weights(labels_csv: str, device: torch.device) -> torch.Tensor:
    """
    @definition : Calcule les pos_weight pour BCEWithLogitsLoss à partir de la
                  fréquence inverse des classes dans train.csv.
                  Chaque poids = neg / pos, clampé à 100 pour éviter les valeurs
                  extrêmes sur les classes très rares.
    @args/params :
        labels_csv (str)           — chemin vers train.csv.
        device     (torch.device)  — device cible pour le tenseur.
    @return : torch.Tensor de shape (12,) contenant les pos_weight.
    """
    df = pd.read_csv(labels_csv)
    weights = []
    logger.info("Poids de classe calculés depuis train.csv :")
    for col in TARGET_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        pos = s.sum()
        neg = len(df) - pos
        w = float(min(neg / pos if pos > 0 else 1.0, 100.0))
        weights.append(w)
        logger.info(
            f"  {col:22s}: pos={int(pos):4d}  neg={int(neg):4d}  weight={w:.2f}"
        )
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    @definition : Entraîne le modèle sur une époque complète.
    @args/params :
        model     — modèle PyTorch.
        loader    — DataLoader d'entraînement.
        optimizer — optimiseur.
        criterion — fonction de perte.
        device    — device cible.
    @return : float — perte moyenne sur l'époque.
    """
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device).float()
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Boucle de validation + collecte des prédictions
# ---------------------------------------------------------------------------


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    @definition : Évalue le modèle sur le jeu de validation.
    @args/params :
        model     — modèle PyTorch en mode eval.
        loader    — DataLoader de validation.
        criterion — fonction de perte.
        device    — device cible.
    @return : tuple (val_loss, all_labels, all_preds) où all_labels et
              all_preds sont des np.ndarray de shape (N, 12).
    """
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val  ", leave=False):
            images = batch["image"].to(device).float()
            labels = batch["labels"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return (
        running_loss / len(loader.dataset),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )


# ---------------------------------------------------------------------------
# Inférence sur le jeu de test
# ---------------------------------------------------------------------------


def predict_test(
    model: nn.Module,
    test_series_csv: str,
    images_dir: str,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[list, np.ndarray]:
    """
    @definition : Génère les prédictions du modèle sur le jeu de test.
                  Charge les séries depuis test_series.csv et retourne les
                  prédictions agrégées par étude (moyenne sur les séries).
    @args/params :
        model           — modèle entraîné en mode eval.
        test_series_csv (str)          — chemin vers test_series.csv.
        images_dir      (str)          — répertoire racine des DICOM de test.
        device          (torch.device) — device cible.
        batch_size      (int)          — taille de batch pour l'inférence.
    @return : tuple (study_ids, predictions) :
              - study_ids   : liste de StudyInstanceUID uniques.
              - predictions : np.ndarray (N_études, 12).
    """
    series_df = pd.read_csv(test_series_csv)

    # Dataset de test : pas de labels
    class TestDataset(Dataset):
        def __init__(self, series_df, images_dir):
            self.data = series_df
            self.images_dir = images_dir

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            series_dir = os.path.join(
                self.images_dir,
                str(row["StudyInstanceUID"]),
                str(row["SeriesInstanceUID"]),
            )
            img = KneeDataset2D._load_central_slice(None, series_dir)
            return img, row["StudyInstanceUID"]

    # Normalisation identique à l'entraînement
    loader = DataLoader(
        TestDataset(series_df, images_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    preds_by_study: dict[str, list] = {}

    with torch.no_grad():
        for images, study_ids in tqdm(loader, desc="Inférence test"):
            images = images.to(device).float()
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            for uid, pred in zip(study_ids, probs):
                preds_by_study.setdefault(uid, []).append(pred)

    # Agrégation : moyenne des prédictions sur toutes les séries d'une étude
    study_ids_sorted = sorted(preds_by_study.keys())
    predictions = np.array(
        [np.mean(preds_by_study[uid], axis=0) for uid in study_ids_sorted],
        dtype=np.float32,
    )

    return study_ids_sorted, predictions


# ---------------------------------------------------------------------------
# Parse des arguments CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    @definition : Parse les arguments de la ligne de commande.
    @args/params : Aucun (lit sys.argv).
    @return : Namespace avec tous les hyperparamètres.
    """
    is_kaggle = os.path.exists("/kaggle/input")
    base = (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection"
        if is_kaggle
        else "./data/raw"
    )
    test_series_default = f"{base}/test_series.csv"
    images_test_default = f"{base}/test_series"

    parser = argparse.ArgumentParser(
        description="Baseline 2D — RSNA Knee Abnormality Detection"
    )
    parser.add_argument("--labels_csv",    default=f"{base}/train.csv")
    parser.add_argument("--series_csv",    default=f"{base}/train_series.csv")
    parser.add_argument("--images_dir",    default=f"{base}/train_series")
    parser.add_argument("--test_series",   default=test_series_default)
    parser.add_argument("--images_test",   default=images_test_default)
    parser.add_argument("--output",        default="/kaggle/working/submission.csv")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--val_split",     type=float, default=0.2)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument(
        "--local_subset",
        type=float,
        default=0.1 if not is_kaggle else 1.0,
        help="Fraction du dataset utilisée en dev local (défaut 10%%).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main() -> None:
    """
    @definition : Pipeline complet : chargement des données, entraînement du
                  modèle baseline 2D avec Weighted BCE, évaluation, inférence
                  sur le jeu de test et génération de submission.csv.
    @args/params : Aucun (utilise parse_args).
    @return : None
    """
    args = parse_args()

    # Reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device : {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = KneeDataset2D(
        labels_csv=args.labels_csv,
        series_csv=args.series_csv,
        images_dir=args.images_dir,
        transform=transform,
    )

    # Sous-ensemble pour le développement local
    if args.local_subset < 1.0:
        n = int(len(full_dataset) * args.local_subset)
        idx = np.random.choice(len(full_dataset), n, replace=False)
        full_dataset = torch.utils.data.Subset(full_dataset, idx)
        logger.info(f"Mode dev local : {n} séries ({args.local_subset*100:.0f}%)")

    # Split train / validation
    val_n = int(len(full_dataset) * args.val_split)
    train_n = len(full_dataset) - val_n
    train_ds, val_ds = random_split(full_dataset, [train_n, val_n])
    logger.info(f"Train : {train_n} séries | Val : {val_n} séries")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # ── Modèle, perte, optimiseur ────────────────────────────────────────────
    model = KneeModel2D(num_classes=12).to(device)
    pos_weights = compute_pos_weights(args.labels_csv, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # ── Entraînement ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_model_path = "best_baseline_2d.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_labels, val_preds = validate(model, val_loader, criterion, device)

        # AUC par classe
        from sklearn.metrics import roc_auc_score
        try:
            aucs = [
                roc_auc_score(val_labels[:, i], val_preds[:, i])
                for i in range(12)
                if val_labels[:, i].sum() > 0
            ]
            mean_auc = float(np.mean(aucs))
        except Exception:
            mean_auc = float("nan")

        logger.info(
            f"Époque {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_AUC={mean_auc:.4f}"
        )

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  → Meilleur modèle sauvegardé ({best_model_path})")

    # ── Inférence sur le jeu de test ─────────────────────────────────────────
    logger.info("Chargement du meilleur modèle pour l'inférence...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    study_ids, predictions = predict_test(
        model,
        test_series_csv=args.test_series,
        images_dir=args.images_test,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Génération de submission.csv ─────────────────────────────────────────
    make_submission(
        study_ids=study_ids,
        predictions=predictions,
        output=args.output,
    )
    logger.info(f"Soumission prête : {args.output}")


if __name__ == "__main__":
    main()
