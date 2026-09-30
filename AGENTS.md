# Agent Guide: RSNA Knee Abnormality Detection

## Project Context : see project_description.md

## data : see https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data 

## models: see https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions or https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/models 

## training code : see https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code 

## evaluation : see https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation 

## submission : see https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/submission


## dev workflow: writes code locally and test on kaggle

## train workflow: kaggle notebook for training and evaluation

## eval workflow: kaggle notebook for evaluation

## submission workflow: kaggle notebook for submission



## Developer Commands
- **Environment**: Managed with `uv`. Use `uv run <command>` for execution.
- **Example of dev command**: Run `python src/data/eda_kaggle.py` to explore dataset distributions and visualize DICOM slices.

## Project Structure & Architecture
- `src/data/`: Data loading and preprocessing.
    - `dataset.py`: PyTorch `KneeDataset` implementation. Handles DICOM $\rightarrow$ Tensor pipeline.
    - `eda_kaggle.py`: Exploration tool for labels and image quality.
- `src/models/`: Model architectures.
- `src/features/`: Feature engineering.
- `data/raw/`: Contains `train.csv` (labels) and `train_series.csv` (series mapping). DICOM files are stored in `train_series/StudyInstanceUID/SeriesInstanceUID/*.dcm`.

## Critical Domain Knowledge
- **DICOM Pipeline**: Always apply `RescaleSlope` and `RescaleIntercept`. Use percentile clipping (1% to 99%) for normalization to handle high-dynamic range outliers.
- **Data Mapping**: Images are organized hierarchically: Study $\rightarrow$ Series $\rightarrow$ Slices. A single `StudyInstanceUID` maps to multiple `SeriesInstanceUID` across different anatomical planes (Sagittal, Axial, Coronal).
- **Labels**: 13 binary labels for various knee abnormalities. Check `train.csv` for the exact list of columns.
