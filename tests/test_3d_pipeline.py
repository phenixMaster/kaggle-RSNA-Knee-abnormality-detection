import torch
from src.data.dataset import KneeDataset
from src.models.knee_model import get_model
import os

def test_3d_pipeline():
    print("Starting 3D Pipeline Test...")
    
    # Configuration minimale pour le test
    labels_csv = "./data/raw/train.csv"
    series_csv = "./data/raw/train_series.csv"
    images_dir = "./data/raw/train_series"
    
    if not os.path.exists(labels_csv):
        print(f"Error: {labels_csv} not found. Please ensure data is in ./data/raw")
        return

    try:
        # 1. Test Dataset
        dataset = KneeDataset(
            labels_csv=labels_csv,
            series_csv=series_csv,
            images_dir=images_dir
        )
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        
        sample = dataset[0]
        img = sample['image']
        labels = sample['labels']
        
        print(f"Sample image shape: {img.shape}") # Expected: (1, 32, 384, 384)
        print(f"Sample labels shape: {labels.shape}") # Expected: (12,)
        
        # 2. Test Model
        model = get_model(model_type='3D')
        model.eval()
        
        with torch.no_grad():
            # Add batch dimension: (1, 1, 32, 384, 384)
            input_tensor = img.unsqueeze(0) 
            output = model(input_tensor)
            
        print(f"Model output shape: {output.shape}") # Expected: (1, 12)
        
        # Verifications
        assert img.shape == (1, 32, 384, 384), f"Wrong image shape: {img.shape}"
        assert output.shape == (1, 12), f"Wrong output shape: {output.shape}"
        
        print("\n✅ 3D Pipeline Test Passed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3d_pipeline()
