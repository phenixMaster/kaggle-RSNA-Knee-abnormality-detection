import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split
from src.data.dataset import KneeDataset
from src.models.knee_model import get_model
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.metrics import calculate_auc

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device).float()
        labels = batch['labels'].to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        # Convert to float32 and store only necessary info to save RAM
        all_labels.append(labels.detach().cpu().numpy().astype(np.float32))
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy().astype(np.float32))
        
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader.dataset), np.concatenate(all_labels), np.concatenate(all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device).float()
            labels = batch['labels'].to(device).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            
    return running_loss / len(loader.dataset), np.concatenate(all_labels), np.concatenate(all_preds)

def main():
    # Configuration
    is_kaggle = os.path.exists("/kaggle/input")
    base_path = "/kaggle/input/competitions/rsna-knee-abnormality-detection" if is_kaggle else "./data/raw"
    
    # Device selection: CUDA for Kaggle, MPS for Mac, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    CONFIG = {
        "labels_csv": os.path.join(base_path, "train.csv"),
        "series_csv": os.path.join(base_path, "train_series.csv"),
        "images_dir": os.path.join(base_path, "train_series"),
        "batch_size": 16,
        "lr": 1e-4,
        "epochs": 8,
        "device": device,
        "val_split": 0.2,
        "local_subset": 0.1 if not is_kaggle else 1.0 # Use 10% of data locally to speed up dev
    }
    
    print(f"Using device: {CONFIG['device']}")
    
    # Dataset & Transforms
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])
    
    full_dataset = KneeDataset(
        labels_csv=CONFIG["labels_csv"],
        series_csv=CONFIG["series_csv"],
        images_dir=CONFIG["images_dir"],
        transform=transform
    )

    # Use a subset of data for local development
    if CONFIG["local_subset"] < 1.0:
        print(f"Local dev mode: using {CONFIG['local_subset']*100:.0f}% of the dataset")
        subset_size = int(len(full_dataset) * CONFIG["local_subset"])
        indices = np.random.choice(len(full_dataset), subset_size, replace=False)
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split Train/Val
    val_size = int(len(full_dataset) * CONFIG["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Use pin_memory=True only for CUDA devices
    use_pin_memory = (CONFIG["device"].type == "cuda")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=use_pin_memory)
    
    # Model, Loss, Optimizer
    model = get_model().to(CONFIG["device"])

    # Progressive Fine-tuning: Freeze backbone for the first few epochs
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    for param in actual_model.backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze the custom classifier
    for param in actual_model.backbone.classifier.parameters():
        param.requires_grad = True

    # Calculate pos_weights for Weighted BCE Loss
    train_df = pd.read_csv(CONFIG["labels_csv"])
    # Use the same logic as KneeDataset to identify target columns
    target_cols = [col for col in train_df.columns if col not in ["StudyInstanceUID", "Report"]]
    
    pos_weights = []
    print("\nCalculating class weights:")
    for col in target_cols:
        # Ensure the column is numeric to avoid TypeError with string labels
        col_data = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        pos = col_data.sum()
        neg = len(train_df) - pos
        weight = neg / pos if pos > 0 else 1.0
        pos_weights.append(weight)
        print(f"{col}: pos={int(pos)}, neg={int(neg)}, weight={weight:.4f}")
    
    weights_tensor = torch.tensor(pos_weights, dtype=torch.float).to(CONFIG["device"])
    
    # Utilisation de DataParallel si plusieurs GPUs sont disponibles
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    
    best_val_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Progressive Fine-tuning: Unfreeze backbone after 2 epochs
        if epoch == 2:
            print("Unfreezing backbone for full fine-tuning...")
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            for param in actual_model.backbone.parameters():
                param.requires_grad = True
        
        train_loss, train_labels, train_preds = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG["device"])
        val_loss, val_labels, val_preds = validate(model, val_loader, criterion, CONFIG["device"])
        
        mean_auc, class_aucs = calculate_auc(val_labels, val_preds)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(mean_auc)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUC: {mean_auc:.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

    # Plotting the results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_auc"], label="Val AUC", color='green')
    plt.title("Validation AUC-ROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close('all')
    print("\nTraining curves saved as training_curves.png")
    
    print(f"\nTraining complete. Best Val AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()
