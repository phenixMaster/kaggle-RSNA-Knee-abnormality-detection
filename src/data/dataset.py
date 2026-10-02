import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from src.utils.image_utils import load_dicom_image

class KneeDataset(Dataset):
    """
    Dataset PyTorch pour charger les images DICOM et les labels associés.
    Ce dataset mappe chaque série d'images à ses labels d'étude correspondants.
    """
    
    def __init__(self, labels_csv: str, series_csv: str, images_dir: str, transform=None):
        """
        Initialise le dataset en fusionnant les fichiers de labels et de séries.
        
        Args:
            labels_csv (str): Chemin vers train.csv (StudyInstanceUID, labels).
            series_csv (str): Chemin vers train_series.csv (StudyInstanceUID, SeriesInstanceUID).
            images_dir (str): Racine du répertoire contenant les dossiers d'images DICOM.
            transform (callable, optional): Transformations PyTorch à appliquer.
        """
        labels_df = pd.read_csv(labels_csv)
        series_df = pd.read_csv(series_csv)
        
        # Jointure : chaque ligne représente maintenant une série unique liée à ses labels d'étude
        self.data_info = pd.merge(series_df, labels_df, on='StudyInstanceUID')
        self.images_dir = images_dir
        self.transform = transform
        
        # Identification dynamique des colonnes de labels (exclut les métadonnées)
        actual_labels = [col for col in labels_df.columns if col not in ['StudyInstanceUID', 'Report']]
        self.label_columns = actual_labels
        print(f"Dataset labels found: {self.label_columns}")

    def __len__(self):
        """Renvoie le nombre total de séries disponibles."""
        return len(self.data_info)

    def __getitem__(self, idx: int):
        """
        Charge plusieurs coupes consécutives d'une série pour une approche 2.5D.
        
        Args:
            idx (int): Index de l'échantillon.
        Returns:
            dict: {'image': Tensor (3, 384, 384), 'labels': Tensor (12,)}
        """
        row = self.data_info.iloc[idx]
        study_id = row['StudyInstanceUID']
        series_id = row['SeriesInstanceUID']
        
        # Chemin : root/StudyInstanceUID/SeriesInstanceUID/
        series_dir = os.path.join(self.images_dir, str(study_id), str(series_id))
        
        images_list = []
        if os.path.exists(series_dir):
            files = sorted([f for f in os.listdir(series_dir) if f.endswith('.dcm')])
            if files:
                # Sélection d'un index aléatoire et prise de 3 coupes consécutives
                idx_start = np.random.randint(0, max(1, len(files) - 2))
                for i in range(3):
                    if idx_start + i < len(files):
                        dicom_path = os.path.join(series_dir, files[idx_start + i])
                        img = load_dicom_image(dicom_path)
                        if img is not None:
                            img = torch.from_numpy(img).float().unsqueeze(0)
                            img = T.functional.resize(img, (384, 384), interpolation=T.InterpolationMode.BICUBIC)
                            images_list.append(img)
                        else:
                            images_list.append(torch.zeros((1, 384, 384)))
                    else:
                        images_list.append(torch.zeros((1, 384, 384)))
        
        if not images_list:
            # Fallback : 3 images noires
            images_list = [torch.zeros((1, 384, 384)) for _ in range(3)]
        
        # Stack des 3 coupes pour créer un tenseur (3, 384, 384)
        image = torch.cat(images_list, dim=0)
        
        if self.transform:
            image = self.transform(image)
            
        # Préparation des labels : conversion float et gestion des NaN
        labels = row[self.label_columns].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=0.0)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels
        }

