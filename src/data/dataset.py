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
    
    def __init__(self, labels_csv: str, series_csv: str, images_dir: str, text_labels_csv: str = None, transform=None):
        """
        Initialise le dataset en fusionnant les fichiers de labels et de séries.
        
        Args:
            labels_csv (str): Chemin vers train.csv (StudyInstanceUID, labels).
            series_csv (str): Chemin vers train_series.csv (StudyInstanceUID, SeriesInstanceUID).
            images_dir (str): Racine du répertoire contenant les dossiers d'images DICOM.
            text_labels_csv (str, optional): Chemin vers train_text_labels.csv (scores de Gemma).
            transform (callable, optional): Transformations PyTorch à appliquer.
        """
        labels_df = pd.read_csv(labels_csv)
        series_df = pd.read_csv(series_csv)
        
        # Jointure : chaque ligne représente maintenant une série unique liée à ses labels d'étude
        self.data_info = pd.merge(series_df, labels_df, on='StudyInstanceUID')
        
        # Fusion avec les labels textuels si fournis
        if text_labels_csv and os.path.exists(text_labels_csv):
            text_df = pd.read_csv(text_labels_csv)
            self.data_info = pd.merge(self.data_info, text_df, on='StudyInstanceUID', suffixes=('', '_text'))
            print(f"Successfully merged text labels from {text_labels_csv}")
        
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
        Charge un volume de coupes d'une série pour une approche 3D.
        
        Args:
            idx (int): Index de l'échantillon.
        Returns:
            dict: {'image': Tensor (1, depth, 384, 384), 'labels': Tensor (12,)}
        """
        row = self.data_info.iloc[idx]
        study_id = row['StudyInstanceUID']
        series_id = row['SeriesInstanceUID']
        
        series_dir = os.path.join(self.images_dir, str(study_id), str(series_id))
        
        depth_target = 32 # Nombre de coupes cible pour le volume 3D
        images_list = []
        
        if os.path.exists(series_dir):
            files = sorted([f for f in os.listdir(series_dir) if f.endswith('.dcm')])
            if files:
                # Échantillonnage pour atteindre depth_target
                num_files = len(files)
                if num_files >= depth_target:
                    # Échantillonnage uniforme si trop de coupes
                    indices = np.linspace(0, num_files - 1, depth_target, dtype=int)
                else:
                    # Remplissage avec padding ou répétition si pas assez de coupes
                    indices = np.concatenate([
                        np.arange(num_files), 
                        np.full(depth_target - num_files, num_files - 1)
                    ])
                
                for i in indices:
                    dicom_path = os.path.join(series_dir, files[i])
                    img = load_dicom_image(dicom_path)
                    if img is not None:
                        img = torch.from_numpy(img).float().unsqueeze(0)
                        img = T.functional.resize(img, (384, 384), interpolation=T.InterpolationMode.BICUBIC)
                        images_list.append(img)
                    else:
                        images_list.append(torch.zeros((1, 384, 384)))
        
        if not images_list:
            images_list = [torch.zeros((1, 384, 384)) for _ in range(depth_target)]
        elif len(images_list) < depth_target:
            # Fallback security padding
            images_list.extend([torch.zeros((1, 384, 384)) for _ in range(depth_target - len(images_list))])
        
        # Stack des coupes pour créer un volume (depth, 1, 384, 384) -> (1, depth, 384, 384)
        image = torch.stack(images_list, dim=0).squeeze(1) # (depth, 384, 384)
        image = image.unsqueeze(0) # (1, depth, 384, 384)
        
        if self.transform:
            image = self.transform(image)
            
        # Préparation des labels : fusion binaire + textuelle (Soft Labels)
        binary_labels = row[self.label_columns].values.astype(np.float32)
        
        # On cherche les colonnes correspondantes dans le DataFrame fusionné
        # Si on a fusionné avec text_labels, les colonnes s'appellent 'Label' et 'Label_text'
        text_labels = []
        for col in self.label_columns:
            text_col = f"{col}_text"
            if text_col in row:
                text_labels.append(row[text_col])
            else:
                text_labels.append(binary_labels[self.label_columns.index(col)])
        
        text_labels = np.array(text_labels).astype(np.float32)
        
        # Target = (Binaire + Texte) / 2
        labels = (binary_labels + text_labels) / 2.0
        labels = np.nan_to_num(labels, nan=0.0)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels
        }

