# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import IntrusionDetectionModel
from config import Config
import os

# Simulated dataset for network traffic sequences
class NetworkTrafficDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=50, input_dim=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            if np.random.rand() > 0.95:
                sequence = np.random.randn(seq_len, input_dim) + np.random.uniform(5, 10)
                label = 1  # intrusion
            else:
                sequence = np.random.randn(seq_len, input_dim)
                label = 0  # normal traffic
            self.data.append(sequence.astype(np.float32))
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(config, epochs=50, batch_size=64):
    dataset = NetworkTrafficDataset(num_samples=50000, seq_len=config.SEQ_LEN, input_dim=config.INPUT_DIM)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model_config = {
        'MODEL_TYPE': config.MODEL_TYPE,
        'INPUT_DIM': config.INPUT_DIM,
        'HIDDEN_DIM': config.HIDDEN_DIM,
        'NUM_LAYERS': config.NUM_LAYERS,
        'BIDIRECTIONAL': config.BIDIRECTIONAL,
        'D_MODEL': config.D_MODEL,
        'NHEAD': config.NHEAD,
        'NUM_CLASSES': config.NUM_CLASSES,
        'LATENT_DIM': config.LATENT_DIM,
        'DEVICE': config.DEVICE
    }
    model = IntrusionDetectionModel(model_config)
    model.to(config.DEVICE)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(config.DEVICE)
            labels = torch.tensor(labels).to(config.DEVICE).long()
            
            optimizer.zero_grad()
            logits, reconstruction, features = model(sequences)
            loss_cls = criterion_cls(logits, labels)
            loss_rec = criterion_rec(reconstruction, features.detach())
            loss = loss_cls + config.REC_WEIGHT * loss_rec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        scheduler.step(running_loss)
        
    backend_dir = "backend"
    if not os.path.exists(backend_dir):
        print("⚠️ 'backend/' directory not found. Creating 'new_dir/' instead...")
        backend_dir = "new_dir"
        os.makedirs(backend_dir, exist_ok=True)
    else:
        print("✅ Using existing 'backend/' directory.")
    
    model_save_path = os.path.join("backend", "intrusion_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved at: {model_save_path}")

if __name__ == '__main__':
    train_model(Config, epochs=30, batch_size=128)
