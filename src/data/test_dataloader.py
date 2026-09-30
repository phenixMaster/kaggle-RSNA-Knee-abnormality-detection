import torch
from torch.utils.data import DataLoader
from src.data.dataset import KneeDataset
import torchvision.transforms as T

def get_knee_dataloader(labels_csv, series_csv, images_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Crée un DataLoader pour le KneeDataset avec batching et parallélisme.
    """
    
    # Transformations recommandées pour l'entraînement
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485], std=[0.229]) # Normalisation ImageNet (adaptée 1 canal)
    ])

    dataset = KneeDataset(
        labels_csv=labels_csv,
        series_csv=series_csv,
        images_dir=images_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True # Accélère le transfert vers le GPU
    )

    return dataloader

if __name__ == "__main__":
    # Test rapide du DataLoader
    try:
        loader = get_knee_dataloader(
            labels_csv="./data/raw/train.csv",
            series_csv="./data/raw/train_series.csv",
            images_dir="./data/raw/train_series"
        )
        
        batch = next(iter(loader))
        print(f"Batch images shape: {batch['image'].shape}") # [batch_size, 1, 224, 224]
        print(f"Batch labels shape: {batch['labels'].shape}") # [batch_size, 13]
        print("DataLoader opérationnel !")
    except Exception as e:
        print(f"Erreur lors du test du DataLoader: {e}")
