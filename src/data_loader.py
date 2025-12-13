import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class StudentDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

# --- THIS WAS MISSING ---
def load_and_process_data(filepath):
    # Load Data
    df = pd.read_csv(filepath)
    
    # Drop ID if exists
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
    
    # Separate Target
    if 'exam_score' in df.columns:
        X = df.drop(columns=['exam_score'])
        y = df['exam_score']
    else:
        X = df
        y = None

    # Encode Categorical Data
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def get_dataloaders(filepath, batch_size=32):
    # Use the helper function above
    X, y, scaler = load_and_process_data(filepath)
    
    # Split
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    # Return Datasets (For Autoencoder we only use X)
    train_loader = DataLoader(StudentDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StudentDataset(X_val), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X.shape[1]