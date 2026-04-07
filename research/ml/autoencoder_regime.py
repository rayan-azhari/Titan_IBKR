"""autoencoder_regime.py -- Unsupervised regime discovery via autoencoder.

Trains an autoencoder on the 72-feature set to learn compressed market state
representations. The bottleneck embedding is clustered (KMeans) to discover
regimes that the rule-based SMA(50/200) detector misses.

Usage in stacking pipeline:
    1. Train autoencoder on IS features (unsupervised)
    2. Extract regime features (embeddings + cluster one-hots)
    3. Append to original features (72 -> 84)
    4. Run stacking on augmented features

Architecture:
    Encoder: 72 -> 32 (ReLU) -> 8 (latent)
    Decoder: 8 -> 32 (ReLU) -> 72 (reconstruction)
    Loss: MSE
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


class MarketAutoencoder(nn.Module):
    """Simple autoencoder for market regime discovery."""

    def __init__(self, input_dim: int = 72, latent_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 8,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
) -> MarketAutoencoder:
    """Train autoencoder on feature matrix (unsupervised).

    Args:
        X: (n_samples, n_features) feature matrix, NaN-cleaned
        latent_dim: bottleneck dimension
        epochs: max training epochs
        batch_size: mini-batch size
        lr: learning rate
        patience: early stopping patience
    """
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n = len(X_clean)
    val_start = int(n * 0.85)

    X_tr = torch.tensor(X_clean[:val_start])
    X_val = torch.tensor(X_clean[val_start:])

    input_dim = X_clean.shape[1]
    model = MarketAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(X_tr, X_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, _ in train_loader:
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon, _ = model(X_val)
            val_loss = criterion(val_recon, X_val).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def extract_regime_features(
    model: MarketAutoencoder,
    X: np.ndarray,
    n_clusters: int = 4,
    kmeans_model: KMeans | None = None,
) -> tuple[np.ndarray, KMeans]:
    """Extract regime features from trained autoencoder.

    Returns:
        features: (n, n_clusters + latent_dim) array
            - First n_clusters columns: one-hot cluster membership
            - Next latent_dim columns: raw embedding values
        kmeans: fitted KMeans model (pass to OOS extraction)
    """
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_tensor = torch.tensor(X_clean)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(X_tensor).cpu().numpy()

    # Cluster embeddings
    if kmeans_model is None:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_model.fit(embeddings)

    cluster_ids = kmeans_model.predict(embeddings)

    # One-hot encode clusters
    one_hot = np.zeros((len(X), n_clusters), dtype=np.float32)
    one_hot[np.arange(len(X)), cluster_ids] = 1.0

    # Combine: one-hot clusters + raw embeddings
    features = np.hstack([one_hot, embeddings.astype(np.float32)])

    return features, kmeans_model
