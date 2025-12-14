import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import DenoisingAutoencoder
import os

DATA_PATH = "data/raw/student_habits_performance.csv"
MODEL_PATH = "outputs/models/autoencoder.pth"

from src.utils import seed_everything
seed_everything(42)

def train():
    train_loader, val_loader, input_dim = get_dataloaders(DATA_PATH)
    
    model = DenoisingAutoencoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Starting Training")
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            noise = torch.randn_like(batch) * 0.1
            noisy_batch = batch + noise
            
            optimizer.zero_grad()
            reconstructed, _ = model(noisy_batch)
            
            loss = criterion(reconstructed, batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()