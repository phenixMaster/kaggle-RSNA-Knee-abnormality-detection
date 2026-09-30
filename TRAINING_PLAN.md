# Training Plan: RSNA Knee Abnormality Detection

## Goals
Implement a complete training pipeline to detect 13 knee abnormalities from DICOM images.

## Execution Steps

### 1. Model Implementation (`src/models/knee_model.py`)
- **Architecture**: Use a pre-trained backbone (e.g., EfficientNet or ResNet) adapted for single-channel (grayscale) input.
- **Head**: Multi-label classification head with a linear layer outputting 13 values.
- **Activation**: Sigmoid activation for each label to handle independent binary classifications.

### 2. Training Orchestration (`main.py`)
- **Data Pipeline**: 
    - Instantiate `KneeDataset`.
    - Use `DataLoader` for batching and multi-processing.
- **Optimization**:
    - **Loss Function**: Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`).
    - **Optimizer**: Adam or AdamW with a learning rate scheduler.
- **Loop**: Standard PyTorch training loop with epoch-based logging.

### 3. Validation Strategy
- **Split**: Implement a stratified split of `train.csv` to ensure class distribution is preserved in the validation set.
- **Metrics**: 
    - AUC-ROC per label.
    - Mean AUC across all labels.
- **Verification**: Save the best model based on validation AUC.
