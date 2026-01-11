import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import DenoisingAutoencoder
import os

DATA_PATH = "data/raw/student_habits_performance.csv"
MODEL_PATH = "outputs/models/autoencoder.pth"

from utils import seed_everything
seed_everything(42)

def train():
    train_loader, val_loader, input_dim = get_dataloaders(DATA_PATH)
    
    model = DenoisingAutoencoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    print("Starting Training")
    
    for epoch in range(500):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            noise = torch.randn_like(batch) * 0.1
            noisy_batch = batch + noise
            
            optimizer.zero_grad()
            reconstructed, _ = model(noisy_batch)
            
            loss = criterion(reconstructed, batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()   

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                reconstructed, _ = model(batch) 
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()