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
        
        if 'debug_done' not in locals():
            # DEBUG: Vérifier les prédictions au tout premier batch de l'entraînement
            with torch.no_grad():
                debug_preds = torch.sigmoid(model(images))
                print(f"\n--- DEBUG TRAIN BATCH 0 ---")
                print(f"Preds mean: {debug_preds.mean().item():.4f} | std: {debug_preds.std().item():.4f}")
                print(f"Labels mean: {labels.mean().item():.4f}")
                print(f"Max pred: {debug_preds.max().item():.4f} | Min pred: {debug_preds.min().item():.4f}")
                print(f"---------------------------\n")
            debug_done = True
            
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
        for i, batch in enumerate(loader):
            images = batch['image'].to(device).float()
            labels = batch['labels'].to(device).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # DEBUG: Afficher les stats du premier batch
            if i == 0:
                preds = torch.sigmoid(outputs)
                print(f"\n--- DEBUG BATCH 0 ---")
                print(f"Preds mean: {preds.mean().item():.4f} | std: {preds.std().item():.4f}")
                print(f"Labels mean: {labels.mean().item():.4f}")
                print(f"Max pred: {preds.max().item():.4f} | Min pred: {preds.min().item():.4f}")
                print(f"---------------------\n")
                break
            
            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            
    return running_loss / len(loader.dataset), np.concatenate(all_labels), np.concatenate(all_preds)

def main():
    # Configuration
    is_kaggle = os.path.exists("/kaggle/input")
    if is_kaggle:
        base_path = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    else:
        base_path = "./data/raw"
    
    # Force images_dir to the exact path provided by the user for Kaggle
    if is_kaggle:
        images_dir = os.path.join(base_path, "train_series")
    else:
        images_dir = os.path.join(base_path, "train_series")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    CONFIG = {
        "labels_csv": os.path.join(base_path, "train.csv"),
        "series_csv": os.path.join(base_path, "train_series.csv"),
        "images_dir": images_dir,
        "text_labels_csv": os.path.join(base_path, "../processed/train_text_labels.csv"),
        "batch_size": 16,
        "lr": 5e-4,
        "epochs": 15,
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
    
    print("Initializing Dataset...")
    full_dataset = KneeDataset(
        labels_csv=CONFIG["labels_csv"],
        series_csv=CONFIG["series_csv"],
        images_dir=CONFIG["images_dir"],
        text_labels_csv=CONFIG["text_labels_csv"],
        transform=transform
    )
    print(f"Dataset initialized. Total samples: {len(full_dataset)}")

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
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    
    # Use pin_memory=True only for CUDA devices
    use_pin_memory = (CONFIG["device"].type == "cuda")
    
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    print("DataLoaders created.")
    
    # Model, Loss, Optimizer
    # On entraîne un ensemble de 3 modèles pour moyenner les prédictions (Seed Averaging)
    ensemble_models = []
    seeds = [42, 123, 999]
    
    # Calculate pos_weights for Weighted BCE Loss (Defined once for all models)
    train_df = pd.read_csv(CONFIG["labels_csv"])
    target_cols = [col for col in train_df.columns if col not in ["StudyInstanceUID", "Report"]]
    
    pos_weights = []
    print("\nCalculating class weights:")
    for col in target_cols:
        col_data = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        pos = col_data.sum()
        neg = len(train_df) - pos
        weight = min(neg / pos if pos > 0 else 1.0, 100.0)
        pos_weights.append(weight)
        print(f"{col}: pos={int(pos)}, neg={int(neg)}, weight={weight:.4f}")
    
    weights_tensor = torch.tensor(pos_weights, dtype=torch.float).to(CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
    
    for seed in seeds:
        print(f"Training model with seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = get_model(model_type='2.5D').to(CONFIG["device"])
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        # Training loop for this specific model
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
        
        best_model_auc = 0.0
        for epoch in range(CONFIG["epochs"]):
            train_loss, train_labels, train_preds = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG["device"])
            val_loss, val_labels, val_preds = validate(model, val_loader, criterion, CONFIG["device"])
            
            mean_auc, _ = calculate_auc(val_labels, val_preds)
            if mean_auc > best_model_auc:
                best_model_auc = mean_auc
                torch.save(model.state_dict(), f"best_model_seed_{seed}.pth")
        
        print(f"Model seed {seed} finished. Best Val AUC: {best_model_auc:.4f}")
        ensemble_models.append(best_model_auc)

    final_ensemble_auc = np.mean(ensemble_models)
    print(f"\nFinal Ensemble Mean AUC: {final_ensemble_auc:.4f}")


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
