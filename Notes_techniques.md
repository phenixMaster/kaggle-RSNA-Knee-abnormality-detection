# Notes Techniques - RSNA Knee Abnormality Detection

Ce document sert de journal de bord pour suivre les expérimentations, les modifications architecturales et les résultats obtenus jour après jour.

## Journal des Expériences

### Expérience 1 : Weighted BCE Loss
- **Objectif** : Lutter contre le déséquilibre des classes (certaines anomalies sont très rares).
- **Modifications** :
    - Calcul des `pos_weight` basé sur la fréquence inverse des classes dans `train.csv`.
    - Intégration de `pos_weight` dans `nn.BCEWithLogitsLoss`.
    - Ajout d'un mode `local_subset` (10%) dans `main.py` pour accélérer le développement local.
- **Résultats** :
    - Val Mean AUC : `0.6874`
- **Observations** : Performance stable, proche de la baseline. L'impact du poids des classes est neutre sur l'AUC globale pour le moment.

### Expérience 2 : Optimisation Visuelle (Résolution & 2.5D)
- **Objectif** : Capturer plus de détails et donner une notion de volume au modèle.
- **Modifications** :
    - **Résolution** : Augmentation de la taille d'entrée de $224 \times 224 \to 384 \times 384$ avec interpolation BICUBIC.
    - **Approche 2.5D** : Modification du Dataset pour charger 3 coupes consécutives (stacking) au lieu d'une seule, créant un tenseur de shape `(3, 384, 384)`.
    - **Fine-tuning Progressif** : Gel du backbone pendant les 2 premières époques, puis dégel complet pour affiner les poids.
- **Résultats** : En cours de validation.

### Expérience 3 : Pipeline Multimodal (Gemma 4)
- **Objectif** : Utiliser les rapports de radiologie pour enrichir les labels d'entraînement.
- **Démarche** :
    - **Analyse Textuelle** : Utilisation de `Gemma 2 2B IT` (via `transformers`) pour analyser les rapports et extraire une probabilité (0.0 à 1.0) pour chacune des 12 anomalies.
    - **Optimisation** : Implémentation du batching et du multi-label prompting pour réduire le temps d'inférence de $\sim 69\text{h}$ à $\sim 5\text{h}$.
- **Fusion Multimodale (Soft Labels)** : 
    - Création d'une cible hybride : $\text{Target} = \frac{\text{Label\_Binaire} + \text{Score\_Gemma}}{2}$.
    - Intégration directe dans `KneeDataset` pour fournir des cibles continues au modèle.
- **Observations** : Cette approche permet d'intégrer l'expertise textuelle du radiologue directement dans la fonction de perte, réduisant potentiellement le bruit des labels binaires.

---
