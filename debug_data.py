import os
import pandas as pd
from src.data.dataset import KneeDataset

def debug_data():
    print("--- DATA DEBUG ---")
    labels_csv = "./data/raw/train.csv"
    series_csv = "./data/raw/train_series.csv"
    images_dir = "./data/raw/train_series"
    
    if not os.path.exists(labels_csv):
        print(f"❌ labels_csv NOT FOUND: {labels_csv}")
        return
    
    df_labels = pd.read_csv(labels_csv)
    df_series = pd.read_csv(series_csv)
    print(f"Labels CSV size: {len(df_labels)}")
    print(f"Series CSV size: {len(df_series)}")
    
    try:
        dataset = KneeDataset(labels_csv, series_csv, images_dir)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample image shape: {sample['image'].shape}")
            print(f"Sample labels shape: {sample['labels'].shape}")
            
            # Vérification physique d'un dossier
            row = dataset.data_info.iloc[0]
            path = os.path.join(images_dir, str(row['StudyInstanceUID']), str(row['SeriesInstanceUID']))
            print(f"Checking path: {path}")
            if os.path.exists(path):
                print(f"✅ Path exists. Files: {len(os.listdir(path))}")
            else:
                print(f"❌ Path NOT FOUND")
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")

if __name__ == "__main__":
    debug_data()
