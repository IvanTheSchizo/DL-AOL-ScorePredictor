import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import get_dataloaders
from src.model import StudentPredictor

# CONFIG
DATA_PATH = os.path.join("data", "raw", "student_habits_performance.csv")
MODEL_PATH = os.path.join("outputs", "models", "final_predictor.pth")
OUTPUT_DIR = os.path.join("outputs", "plots")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_correlation_matrix(df):
    """Generates a heatmap of feature correlations."""
    plt.figure(figsize=(10, 8))
    
    # Encode text columns for correlation calculation
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
        
    corr = df_encoded.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Feature Correlation Matrix")
    
    save_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_target_distribution(df):
    """Generates a histogram of the Exam Scores."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df['exam_score'], kde=True, color='teal', bins=20)
    plt.title("Distribution of Student Exam Scores")
    plt.xlabel("Exam Score")
    plt.ylabel("Frequency")
    
    save_path = os.path.join(OUTPUT_DIR, "score_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_actual_vs_predicted():
    """Generates a scatter plot comparing True Scores vs Model Predictions."""
    # 1. Load Validation Data (Features AND Targets)
    _, val_loader, input_dim = get_dataloaders(DATA_PATH, include_targets=True)
    
    # 2. Load Model
    model = StudentPredictor(input_dim=input_dim)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print("❌ Model not found. Skipping prediction plot.")
        return

    model.eval()
    
    # 3. Get Predictions
    actuals = []
    preds = []
    
    with torch.no_grad():
        for X, y in val_loader:
            output = model(X)
            actuals.extend(y.numpy().flatten())
            preds.extend(output.numpy().flatten())
            
    # 4. Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, preds, alpha=0.6, color='purple')
    
    # Draw perfect prediction line
    min_val = min(min(actuals), min(preds))
    max_val = max(max(actuals), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
    
    plt.title("Actual vs Predicted Exam Scores")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    print("🎨 Generating Project Plots...")
    
    # Load Raw Data for EDA
    if os.path.exists(DATA_PATH):
        raw_df = pd.read_csv(DATA_PATH)
        # Drop ID if present
        if 'student_id' in raw_df.columns:
            raw_df = raw_df.drop(columns=['student_id'])
            
        plot_correlation_matrix(raw_df)
        plot_target_distribution(raw_df)
    else:
        print(f"❌ Data not found at {DATA_PATH}")

    # Generate Model Evaluation Plot
    plot_actual_vs_predicted()