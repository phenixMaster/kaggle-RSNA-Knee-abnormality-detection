# Plan d'implémentation : Weighted BCE Loss pour RSNA Knee Detection

## 1. Objectif
Lutter contre le déséquilibre des classes (class imbalance) où les anomalies sont rares par rapport aux cas sains. L'objectif est d'augmenter la sensibilité (recall) du modèle pour les classes positives sans dégrader excessivement la précision.

## 2. Stratégie Technique
Le modèle utilise actuellement `BCEWithLogitsLoss`. Nous allons passer à une version pondérée en utilisant le paramètre `pos_weight`.

### A. Calcul des poids (Inverse Frequency)
Le poids pour chaque classe sera calculé selon la formule :
`pos_weight = nombre_de_negatifs / nombre_de_positifs`

### B. Modifications du Code
1. **Analyse des labels** :
   - Dans `main.py`, analyser `train_df` pour compter les 0 et 1 pour chacune des 12 classes.
   - Générer un tenseur de poids de shape `(12,)`.
2. **Mise à jour du critère** :
   - Initialiser `nn.BCEWithLogitsLoss(pos_weight=weights_tensor)`.
   - Déplacer ce tenseur sur le même `device` que le modèle.

## 3. Workflow d'exécution (Demain)
1. **Phase d'analyse** : Ajouter un script ou une fonction pour imprimer les poids calculés pour chaque anomalie.
2. **Intégration** : Modifier `main.py` pour injecter ces poids dans la perte.
3. **Lancement** : Relancer l'entraînement sur Kaggle.
4. **Évaluation** : Comparer la `Val Mean AUC` et l'AUC par classe avec la baseline actuelle (0.69).

## 4. Indicateurs de Succès
- Augmentation de la `Val Mean AUC`.
- Amélioration significative de l'AUC sur les classes les plus rares.
- Augmentation du nombre de vrais positifs (TP) dans la matrice de confusion.
