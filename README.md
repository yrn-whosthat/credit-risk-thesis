## 📦 Project Overview

This repository contains the full codebase and workflow for my Bachelor's thesis on credit risk modeling. It includes data preprocessing, model training (with SMOTE and Random Forest), evaluation, and visualizations.

## 📁 Project Structure

- `data/` – Dataset (not included in repo)
- `src/` – Scripts for training
- `notebooks/` – Jupyter notebooks for visualization
- `figures/` – Graphs and charts (can be generated)
- `models/` – Trained models (not stored here due to GitHub file limits)

## 🚀 How to Use

1. Upload `credit_risk_dataset.csv` to `data/`
2. (Optional) Run `src/train_model.py` to retrain the model
3. Download pretrained model from link below and place it in root
4. Open and run `notebooks/visualizations.ipynb`

## 🔗 Download Pretrained Model

Due to file size limits, the trained model is hosted externally:  
📥 [Download rf_model.pkl](https://drive.google.com/file/d/160yixnhATwCeIbym8rpSB2gQvQidQbOb/view)

Place the file in the root directory or in `models/` (create the folder if needed).

## ✅ Dependencies

Install all required Python packages:
```bash
pip install -r requirements.txt
