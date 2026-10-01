import pydicom
import numpy as np

def load_dicom_image(file_path: str) -> np.ndarray:
    """
    Charges une image DICOM, applique le rescaling et une normalisation par percentiles.
    """
    try:
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array.astype(np.float32)
        
        # Application des facteurs d'échelle DICOM
        if 'RescaleIntercept' in dicom and 'RescaleSlope' in dicom:
            image = image * dicom.RescaleSlope + dicom.RescaleIntercept
        
        # Normalisation robuste : Clipping des percentiles 1% et 99%
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1 + 1e-8)
        
        return image
    except Exception as e:
        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return None
