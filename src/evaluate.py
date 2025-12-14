import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_loader import get_dataloaders
from model import StudentPredictor
import os

# CONFIG
DATA_PATH = "data/raw/student_habits_performance.csv"
MODEL_PATH = "outputs/models/final_predictor.pth"

def evaluate_model():
    print("--- 📊 Final Model Evaluation ---")
    
    # 1. Load Data (Validation Set)
    # We set include_targets=True so we get (X, y) tuples
    _, val_loader, input_dim = get_dataloaders(DATA_PATH, batch_size=1, include_targets=True)
    
    # 2. Load Model
    model = StudentPredictor(input_dim=input_dim)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print("❌ Model file not found. Run training first.")
        return

    model.eval()
    
    # 3. Inference Loop
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in val_loader:
            y = y.view(-1, 1).float()
            pred = model(X)
            
            all_preds.append(pred.item())
            all_targets.append(y.item())
            
    # 4. Calculate Metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print(f"\nResults on Validation Set ({len(all_targets)} students):")
    print(f"RMSE (Root Mean Sq Error): {rmse:.4f}")
    print(f"MAE  (Mean Abs Error):     {mae:.4f}")
    print(f"R2 Score:                  {r2:.4f}")
    
    # 5. Save Report (Optional but good for 'Final Report' writing)
    results = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2]
    })
    results.to_csv("outputs/evaluation_metrics.csv", index=False)
    print("\n✅ Metrics saved to outputs/evaluation_metrics.csv")

if __name__ == "__main__":
    evaluate_model()