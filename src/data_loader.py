import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class StudentDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Handle targets if they exist
        if targets is not None:
            self.targets = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # If we have targets, return (X, y)
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        # Otherwise just return X (for Autoencoder)
        return self.features[idx]

def load_and_process_data(filepath):
    # Load Data
    df = pd.read_csv(filepath)
    
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

def get_dataloaders(filepath, batch_size=32, include_targets=False):
    X, y, scaler = load_and_process_data(filepath)
    
    # Split (We split both X and y, even if y isn't used later)
    if y is not None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_train, y_val = None, None
    
    # Select mode based on flag
    if include_targets and y is not None:
        train_dataset = StudentDataset(X_train, y_train)
        val_dataset = StudentDataset(X_val, y_val)
    else:
        train_dataset = StudentDataset(X_train, None)
        val_dataset = StudentDataset(X_val, None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X.shape[1]

# --- DATA SAVING FUNCTION (For Repo Requirement) ---
def save_processed_data(filepath, output_dir="data/processed"):
    X, y, scaler = load_and_process_data(filepath)
    
    # Recombine X and y for saving
    df_processed = pd.DataFrame(X)
    if y is not None:
        df_processed['exam_score'] = y.values
        
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "student_data_cleaned.csv")
    df_processed.to_csv(save_path, index=False)
    print(f"✅ Processed data saved to {save_path}")

if __name__ == "__main__":
    # This allows you to run 'python src/data_loader.py' to generate the file
    # We assume the script is run from the project root
    raw_path = os.path.join("data", "raw", "student_habits_performance.csv")
    if os.path.exists(raw_path):
        save_processed_data(raw_path)
    else:
        print(f"❌ Could not find file at {raw_path}")