import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import StudentPredictor
import os

from utils import seed_everything
seed_everything(42)

DATA_PATH = "data/raw/student_habits_performance.csv"
AE_PATH = "outputs/models/autoencoder.pth" 
FINAL_MODEL_PATH = "outputs/models/final_predictor.pth"

def train_predictor():
    from data_loader import load_and_process_data, StudentDataset
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    
    X, y, scaler = load_and_process_data(DATA_PATH)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = StudentDatasetWithTargets(X_train, y_train)
    val_dataset = StudentDatasetWithTargets(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = StudentPredictor(input_dim=X.shape[1], pretrained_ae_path=AE_PATH)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Supervised Training with Attention...")
    
    best_val_loss = float('inf')

    for epoch in range(500):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            y_batch = y_batch.view(-1, 1).float()
            
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss {total_loss:.4f} | Val Loss {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print(f"Final Model saved to {FINAL_MODEL_PATH}")

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_batch = y_batch.view(-1, 1).float()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

from torch.utils.data import Dataset
class StudentDatasetWithTargets(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

if __name__ == "__main__":
    train_predictor()