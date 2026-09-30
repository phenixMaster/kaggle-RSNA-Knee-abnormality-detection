# Plan d'Étude - RSNA Knee Abnormality Detection AI Challenge

## Architecture du Projet
```text
kaggle-RSNA/
├── data/
│   ├── raw/             # Données brutes téléchargées depuis Kaggle (DICOM/PNG)
│   └── processed/       # Données nettoyées, recadrées et transformées (ex: arrays NumPy, HDF5)
├── notebooks/           # Notebooks Jupyter pour l'EDA et les prototypes rapides
├── src/                 # Code source du projet
│   ├── data/            # Scripts de chargement et de préparation des données (dataloaders)
│   ├── features/        # Extraction de caractéristiques et traitement d'image
│   └── models/          # Architectures des modèles, scripts d'entraînement et d'évaluation
├── models/              # Poids des modèles entraînés (.pt, .pth, .onnx)
├── docs/                # Documentation additionnelle
├── pyproject.toml       # Fichier de configuration du projet (généré par uv)
└── PLAN.md              # Ce plan
```

## Étapes de Développement (Workflow)

### 1. Configuration et Préparation (Semaine 1)
- [ ] Mettre à jour le token `kaggle.json` pour autoriser le téléchargement.
- [ ] Télécharger le dataset complet avec l'API Kaggle (`kaggle competitions download -c rsna-knee-abnormality-detection`).
- [ ] Explorer les données (EDA) dans `notebooks/`. Visualiser quelques images DICOM et évaluer la distribution des classes.

### 2. Pipeline de Données (Semaine 2)
- [ ] Écrire les scripts dans `src/data/` pour lire et prétraiter les images DICOM (normalisation, redimensionnement).
- [ ] Gérer les labels faibles/manquants (si applicable) potentiellement via des méthodes NLP sur les comptes-rendus.
- [ ] Mettre en place la validation croisée (ex: GroupKFold sur l'ID du patient).

### 3. Modélisation de Base - 2D / 2.5D (Semaines 3-4)
- [ ] Implémenter une *baseline* solide en 2D (ex: EfficientNet ou ResNet) dans `src/models/`.
- [ ] Entraîner le modèle avec une approche multi-labels (12 classes) en utilisant `BCEWithLogitsLoss`.
- [ ] Enregistrer les expériences et les poids dans `/models/`.

### 4. Modélisation Avancée - 3D (Semaine 5)
- [ ] Passer à des modèles 3D (ex: 3D ResNet) ou 2.5D pour prendre en compte l'aspect volumétrique des IRM.
- [ ] Expérimenter avec l'attention spatiale et la Focal Loss pour les anomalies rares.

### 5. Optimisation (Track d'Efficacité) (Semaine 6)
- [ ] Utiliser des techniques d'optimisation (Mixed Precision, élagage, quantification) pour accélérer le modèle.
- [ ] Vérifier les contraintes de mémoire et de temps pour la soumission Kaggle.

### 6. Ensembling & Soumission Finale (Semaine 7)
- [ ] Combiner les prédictions (Ensembling) des modèles 2D et 3D.
- [ ] Intégrer la Test-Time Augmentation (TTA).
- [ ] Préparer le script final d'inférence pour la soumission sur la plateforme Kaggle.

---
**Rappel de bonnes pratiques :**
- Lancer toutes les commandes Python via `uv run`.
- Installer les dépendances dans l'environnement virtuel avec `uv add <package>`.
- Ajouter des docstrings normées (PEP 8 / PyEPL) à toutes les fonctions dans `src/`.
