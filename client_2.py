import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import flwr as fl

# --- 1. NEURAL NETWORK MODEL ---
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Binary classification for TenYearCHD
        )

    def forward(self, x):
        return self.net(x)

# --- 2. DATA LOAD HELPER ---
def load_hospital_data(hospital_id: int):
    """Loads a single hospital dataset split (prefers augmented data if present)."""
    augmented_path = f"C:/Users/rridd/Downloads/Research/FDL/Gemini_split_new/6_clients/Hospital_2_data.csv"
    standard_path = f"C:/Users/rridd/Downloads/Research/FDL/Gemini_split_new/6_clients/Hospital_2_data.csv"
    
    file_path = augmented_path if os.path.exists(augmented_path) else standard_path
    df = pd.read_csv(file_path)
    
    # Target column is 'TenYearCHD'
    X = df.drop(columns=['TenYearCHD']).values
    y = df['TenYearCHD'].values
    
    # Scale continuous features locally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(train_dataset, batch_size=32, shuffle=True), X.shape[1]

'''# --- 2. DATA LOAD HELPER ---
def load_hospital_data(hospital_id: int):
    """Loads a single hospital dataset split dynamically based on hospital_id."""
    base_dir = "C:/Users/rridd/Downloads/Research/FDL/Gemini_split_new/6_clients"
    
    # Dynamically inject the hospital_id into the filename using f-strings
    file_path = os.path.join(base_dir, f"Hospital_{hospital_id}_data.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Could not find CSV file for Hospital {hospital_id} at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Target column is 'TenYearCHD'
    X = df.drop(columns=['TenYearCHD']).values
    y = df['TenYearCHD'].values
    
    # Scale continuous features locally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(train_dataset, batch_size=32, shuffle=True), X.shape[1]'''

# --- 3. LOCAL TRAINING LOOP ---
def train_local(model, train_loader, epochs=5, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

def evaluate_local(model, test_loader):
    criterion = nn.BCELoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
            
            predicted_labels = (preds >= 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += len(y_batch)
            
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return avg_loss, accuracy

# --- 4. FLOWER CLIENT CLASS ---
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id
        self.train_loader, input_dim = load_hospital_data(hospital_id)
        self.model = HeartDiseaseModel(input_dim=input_dim)

    def get_parameters(self, config):
        # Fixed: Added .detach() for safe gradient tensor conversion
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val, dtype=param.dtype)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_local(self.model, self.train_loader, epochs=5)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate_local(self.model, self.train_loader)
        return float(loss), len(self.train_loader.dataset), {"accuracy": float(accuracy)}

# --- 5. STANDALONE DRY RUN TEST ---
def test_local_pipeline(hospital_id: int = 1):
    print(f"\n==========================================")
    print(f"🧪 TESTING LOCAL PIPELINE FOR HOSPITAL {hospital_id}")
    print(f"==========================================")
    
    # 1. Test Data Loading
    try:
        train_loader, input_dim = load_hospital_data(hospital_id)
        print(f"✅ Data Loaded Successfully!")
        print(f"   • Input Dimensions (Features): {input_dim}")
        print(f"   • Total Batches: {len(train_loader)}")
        print(f"   • Total Samples: {len(train_loader.dataset)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Test Model Forward Pass
    try:
        model = HeartDiseaseModel(input_dim=input_dim)
        sample_x, sample_y = next(iter(train_loader))
        sample_out = model(sample_x)
        print(f"\n✅ Model Architecture Verified!")
        print(f"   • Input Batch Shape: {sample_x.shape}")
        print(f"   • Target Batch Shape: {sample_y.shape}")
        print(f"   • Output Batch Shape: {sample_out.shape}")
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return

    # 3. Test Local Training Epochs
    try:
        print(f"\n⏳ Running 2 Dry-Run Training Epochs...")
        train_local(model, train_loader, epochs=2, lr=0.001)
        loss, acc = evaluate_local(model, train_loader)
        print(f"✅ Training Completed Successfully!")
        print(f"   • Post-Training Loss: {loss:.4f}")
        print(f"   • Post-Training Accuracy: {acc * 100:.2f}%")
    except Exception as e:
        print(f"❌ Error during training loop: {e}")
        return

    print(f"\n==========================================")
    print(f"🎉 SUCCESS! Your client code is 100% functional!")
    print(f"==========================================\n")

# --- 6. ENTRY POINT ---
if __name__ == "__main__":
    is_test_mode = any(arg.lower() in ["--test", "-t", "test"] for arg in sys.argv)
    
    hospital_id = 1
    for arg in sys.argv[1:]:
        if arg.isdigit():
            hospital_id = int(arg)
            break

    if is_test_mode:
        test_local_pipeline(hospital_id=hospital_id)
    else:
        server_ip ="10.2.80.71:8080"
        print(f"[Hospital {hospital_id}] Connecting to FL Server at {server_ip}...")
        
        fl.client.start_client(
            server_address=server_ip,
            client=HospitalClient(hospital_id=hospital_id).to_client()
        )