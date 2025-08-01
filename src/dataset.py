# src/dataset.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_dataset(path="data/processed/features.csv", test_size=0.2, batch_size=32, seed=42):
    # Load feature matrix
    features = pd.read_csv(path).values.astype(np.float32)

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train/test split
    X_train, X_test = train_test_split(
        features_scaled, test_size=test_size, random_state=seed
    )

    # Create PyTorch Datasets
    train_dataset = TensorDataset(torch.tensor(X_train))
    test_dataset = TensorDataset(torch.tensor(X_test))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, scaler

