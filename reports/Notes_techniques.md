# Notes Techniques - RSNA Knee Abnormality Detection

Ce document sert de journal de bord pour suivre les expérimentations, les modifications architecturales et les résultats obtenus jour après jour.

## Journal des Expériences

### [Date du jour] - Implémentation de la Weighted BCE Loss
- **Objectif** : Lutter contre le déséquilibre des classes (certaines anomalies sont très rares).
- **Modifications** :
    - Calcul des `pos_weight` basé sur la fréquence inverse des classes dans `train.csv`.
    - Intégration de `pos_weight` dans `nn.BCEWithLogitsLoss`.
    - Ajout d'un mode `local_subset` (10%) dans `main.py` pour accélérer le développement local.
- **Résultats** :
    - Val Mean AUC : `0.6874`
- **Observations** : Performance stable, proche de la baseline. L'impact du poids des classes est neutre sur l'AUC globale pour le moment.

### [Date du jour] - Nouvel entraînement
- **Résultats** :
    - Val Mean AUC : `0.67`
- **Observations** : Légère baisse de performance par rapport à l'expérience précédente.

---
