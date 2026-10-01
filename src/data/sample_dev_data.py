import os
import pandas as pd
import shutil
import numpy as np

def sample_dataset(raw_dir, processed_dir, sample_fraction=0.01):
    """
    Télécharge une fraction aléatoire des données pour le développement local.
    Gère les structures de dossiers Kaggle et locales simplifiées.
    """
    print(f"Échantillonnage du dataset ({sample_fraction*100}%)...")
    
    series_csv = os.path.join(raw_dir, "train_series.csv")
    if not os.path.exists(series_csv):
        print(f"Erreur : {series_csv} introuvable.")
        return

    df_series = pd.read_csv(series_csv)
    sampled_series = df_series.sample(frac=sample_fraction, random_state=42)
    
    os.makedirs(processed_dir, exist_ok=True)
    
    count = 0
    for _, row in sampled_series.iterrows():
        study_id = row['StudyInstanceUID']
        series_id = row['SeriesInstanceUID']
        
        # Tentative 1 : Structure Kaggle (train_series/study/series)
        src_dir = os.path.join(raw_dir, "train_series", study_id, series_id)
        # Tentative 2 : Structure locale simplifiée (directement dans raw)
        if not os.path.exists(src_dir):
            src_dir = os.path.join(raw_dir, study_id, series_id)
        
        if os.path.exists(src_dir):
            dst_dir = os.path.join(processed_dir, study_id, series_id)
            os.makedirs(dst_dir, exist_ok=True)
            for f in os.listdir(src_dir):
                if f.endswith('.dcm'):
                    shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
                    count += 1
        else:
            # Si on ne trouve pas le dossier, on cherche si le fichier DICOM est juste à la racine
            # (Cas où on a quelques fichiers dcm en vrac dans raw)
            for f in os.listdir(raw_dir):
                if f.endswith('.dcm'):
                    dst_dir = os.path.join(processed_dir, "misc")
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(os.path.join(raw_dir, f), os.path.join(dst_dir, f))
                    count += 1
                    break # On n'en prend qu'un pour l'exemple
                    
    shutil.copy2(os.path.join(raw_dir, "train.csv"), os.path.join(processed_dir, "train.csv"))
    shutil.copy2(os.path.join(raw_dir, "train_series.csv"), os.path.join(processed_dir, "train_series.csv"))
    
    print(f"Terminé ! {count} images copiées vers {processed_dir}")

if __name__ == "__main__":
    # Chemins locaux
    RAW_DIR = "./data/raw"
    PROCESSED_DIR = "./data/processed"
    
    sample_dataset(RAW_DIR, PROCESSED_DIR, sample_fraction=0.005) # 0.5% pour commencer léger
