"""
Script EDA pour la compétition Kaggle RSNA Knee Abnormality Detection.
Ce script est conçu pour être exécutable directement sur les Kaggle Notebooks.
"""

import os
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_kaggle_dataset_path() -> str:
    """
    @definition : Détermine le chemin vers le dataset, que ce soit en local ou sur Kaggle.
    @args/params : Aucun
    @return : Chemin du répertoire contenant les données (str).
    """
    kaggle_path = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    local_path = "./data/raw"
    
    if os.path.exists(kaggle_path):
        return kaggle_path
    elif os.path.exists(local_path):
        return local_path
    else:
        return ""

def load_dicom_image(file_path: str) -> np.ndarray:
    """
    @definition : Charge une image DICOM et renvoie son tableau de pixels.
    @args/params : file_path (str) - Le chemin complet vers le fichier .dcm.
    @return : Tableau numpy contenant les pixels de l'image (np.ndarray).
    """
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array
    
    # Transformation des données (si nécessaire)
    if 'RescaleIntercept' in dicom and 'RescaleSlope' in dicom:
        intercept = dicom.RescaleIntercept
        slope = dicom.RescaleSlope
        image = image * slope + intercept
        
    # Normalisation par clipping des percentiles pour un meilleur contraste
    p1, p99 = np.percentile(image, [1, 99])
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)
        
    return image

def plot_dicom_grid(images, titles=None, cols=5):
    """
    @definition : Affiche une grille d'images DICOM.
    @args/params : images (list) - Liste d'images numpy. titles (list) - Liste de titres. cols (int) - Nombre de colonnes.
    @return : None
    """
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='bone')
        if titles:
            plt.title(titles[i], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def explore_dataset_labels(labels_file: str):
    """
    @definition : Charge et affiche des statistiques basiques sur le fichier des labels d'entraînement.
    @args/params : labels_file (str) - Chemin vers le fichier train.csv.
    @return : DataFrame contenant les labels.
    """
    print(f"Chargement des labels depuis : {labels_file}")
    if not os.path.exists(labels_file):
        print("Fichier de labels introuvable.")
        return None
        
    df = pd.read_csv(labels_file)
    
    # Analyse de la distribution des classes
    label_cols = [col for col in df.columns if col not in ['StudyInstanceUID', 'Report']]
    print("\nDistribution des classes :")
    class_counts = df[label_cols].sum().sort_values(ascending=False)
    print(class_counts)
    
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.title("Distribution des anomalies du genou")
    plt.ylabel("Nombre d'occurrences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nAperçu des données :")
    print(df.head())
    
    print("\nInformations générales :")
    print(df.info())
    
    return df

def main():
    """
    @definition : Fonction principale pour exécuter l'exploration de base.
    @args/params : Aucun
    @return : None
    """
    base_path = get_kaggle_dataset_path()
    if not base_path:
        print("Dataset introuvable. Assurez-vous d'être sur Kaggle ou d'avoir les données en local.")
        return

    print(f"Chemin de base utilisé : {base_path}")
    
    # 1. Explorer les labels (train.csv)
    labels_path = os.path.join(base_path, "train.csv")
    labels_df = explore_dataset_labels(labels_path)
    
    # 2. Mapping avec train_series.csv
    series_path = os.path.join(base_path, "train_series.csv")
    if os.path.exists(series_path) and labels_df is not None:
        series_df = pd.read_csv(series_path)
        # Jointure pour avoir labels et metadata des séries
        full_df = pd.merge(labels_df, series_df, on='StudyInstanceUID')
        print("\nMapping réussi : train.csv + train_series.csv")
        print(f"Total de séries : {len(full_df)}")
        
        # Exemple d'affichage ciblé : Première série Sagittale du premier patient
        sample = full_df[full_df['Anatomical_Plane'] == 'Sagittal'].iloc[0]
        study_id = sample['StudyInstanceUID']
        series_id = sample['SeriesInstanceUID']
        
        # Recherche du fichier DICOM
        # Structure attendue : base_path / train_series / study_id / series_id / *.dcm
        train_series_dir = os.path.join(base_path, "train_series")
        series_dir = os.path.join(train_series_dir, study_id, series_id)
        
    # 3. Affichage d'une grille de 10 images aléatoires
    if 'full_df' in locals() and os.path.exists(train_series_dir):
        print("\nChargement de 10 images aléatoires pour la grille...")
        random_samples = full_df.sample(10)
        grid_images = []
        grid_titles = []
        
        for _, row in random_samples.iterrows():
            series_dir = os.path.join(train_series_dir, row['StudyInstanceUID'], row['SeriesInstanceUID'])
            if os.path.exists(series_dir):
                files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
                if files:
                    img_path = os.path.join(series_dir, np.random.choice(files))
                    try:
                        img = load_dicom_image(img_path)
                        grid_images.append(img)
                        grid_titles.append(f"{row['Anatomical_Plane']}\n{row['StudyInstanceUID'][:8]}...")
                    except Exception as e:
                        print(f"Erreur lecture {img_path}: {e}")
        
        if grid_images:
            plot_dicom_grid(grid_images, titles=grid_titles)

    else:
        print("\nFichier train_series.csv manquant ou labels non chargés.")


if __name__ == "__main__":
    main()
