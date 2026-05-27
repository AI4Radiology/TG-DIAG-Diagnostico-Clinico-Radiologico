# notebooks/

This folder contains Jupyter notebooks used for **exploratory data analysis (EDA)**, feature experiments, and model research — the sandbox phase before code is promoted into `src/`.

## Workflow

1. Explore and prototype here freely.
2. Once a step is stable and generalizable, refactor it into `src/` (e.g., `src/preprocess.py`, `src/train.py`).
3. Notebooks should remain reproducible but are **not** part of the production pipeline.

## Contents

| Notebook | Purpose |
|---|---|
| `0.0_Analisis_y_Transformacion.ipynb` | EDA, cleaning, text normalization |
| `0.1_Inferencia_RAD_ALERT.ipynb` | Triage filter validation |
| `1_Modelos_Exploracion.ipynb` | Baseline exploration (SVM, LR, NB, LSTM) |
| `2_Modelo_LSTM_Baseline.ipynb` | Initial BiLSTM trainer |
| `3.0_Balanceo_SMOTE_ROS.ipynb` | Classical oversampling strategies |
| `3.1_Augmentation_LLM.ipynb` | LLM-based clinical text augmentation |
| `4_Modelo_XGBoost_Balanceado.ipynb` | XGBoost on balanced dataset |
| `5_Comparacion_Final.ipynb` | Metrics aggregation and model selection |
| `6_Modelo_Transformers.ipynb` | DistilBERT fine-tuning (GPU) |
