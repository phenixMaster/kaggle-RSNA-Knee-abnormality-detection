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

    def __len__(self):
        """Renvoie le nombre total de séries disponibles."""
        return len(self.data_info)

    def __getitem__(self, idx: int):
        """
        Charge une coupe aléatoire d'une série et ses labels associés.
        
        Args:
            idx (int): Index de l'échantillon.
        Returns:
            dict: {'image': Tensor (1, 224, 224), 'labels': Tensor (13,)}
        """
        row = self.data_info.iloc[idx]
        study_id = row['StudyInstanceUID']
        series_id = row['SeriesInstanceUID']
        
        # Chemin : root/StudyInstanceUID/SeriesInstanceUID/
        series_dir = os.path.join(self.images_dir, str(study_id), str(series_id))
        
        # Initialisation avec une image noire en cas d'échec de lecture
        # Note: la taille ici est une sécurité, le resize final garantit la dimension
        img_array = np.zeros((224, 224), dtype=np.float32)
        
        if os.path.exists(series_dir):
            files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
            if files:
                    # Sélection aléatoire d'une coupe pour augmenter la diversité d'entraînement
                    dicom_path = os.path.join(series_dir, np.random.choice(files))
                    img_array = load_dicom_image(dicom_path)
                    if img_array is None:
                        pass
        
        # Conversion en Tensor PyTorch (Channel, Height, Width)
        image = torch.from_numpy(img_array).unsqueeze(0) 
        
        # FORCE LE REDIMENSIONNEMENT ICI pour éviter le RuntimeError: stack expects each tensor to be equal size
        # On le fait AVANT le transform optionnel
        image = T.functional.resize(image, (224, 224))
        
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

