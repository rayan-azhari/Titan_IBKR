"""lstm_features.py -- Phase 0C: LSTM Temporal Feature Extractor.

Trains a 2-layer LSTM (64 hidden) on a lookback window of base features,
extracts hidden states as temporal features for XGBoost. The LSTM learns
temporal patterns that XGBoost cannot (sequence dependencies, regime transitions).

Architecture:
    Input:  (batch, lookback=20, n_features)
    LSTM:   2 layers, 64 hidden, dropout=0.2
    Output: last hidden state → 64-dim feature vector per bar

Training:
    - Trained per WFO fold on IS data only (no lookahead)
    - Binary cross-entropy on same regime+pullback labels
    - Early stopping on validation loss (last 20% of IS)

Features produced:
    lstm_h_{0..63}: 64 hidden state features per bar
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMFeatureExtractor(nn.Module):
    """2-layer LSTM for temporal feature extraction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, hidden_state)."""
        # x: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n: (n_layers, batch, hidden) → last layer
        last_hidden = h_n[-1]  # (batch, hidden)
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits, last_hidden

    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """Extract hidden state features (no grad)."""
        self.eval()
        with torch.no_grad():
            _, hidden = self.forward(x)
        return hidden.cpu().numpy()


def build_sequences(
    X: np.ndarray,
    y: np.ndarray | None = None,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build (batch, lookback, features) sequences from flat feature matrix.

    Each sequence ends at bar t, containing bars [t-lookback+1, ..., t].
    Labels (if provided) correspond to bar t.
    """
    n, d = X.shape
    if n < lookback:
        raise ValueError(f"Not enough data: {n} < lookback {lookback}")

    sequences = np.zeros((n - lookback + 1, lookback, d), dtype=np.float32)
    for i in range(n - lookback + 1):
        sequences[i] = X[i:i + lookback]

    labels = None
    if y is not None:
        labels = y[lookback - 1:]

    return sequences, labels


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int = 20,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
) -> LSTMFeatureExtractor:
    """Train LSTM on IS data with early stopping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build sequences
    seqs, labels = build_sequences(X_train, y_train, lookback)
    if labels is None or len(seqs) < 100:
        raise ValueError(f"Too few sequences: {len(seqs)}")

    # Train/val split (last 20% of IS for early stopping)
    n = len(seqs)
    val_start = int(n * 0.8)
    X_tr = torch.tensor(seqs[:val_start], dtype=torch.float32)
    y_tr = torch.tensor(labels[:val_start], dtype=torch.float32)
    X_val = torch.tensor(seqs[val_start:], dtype=torch.float32)
    y_val = torch.tensor(labels[val_start:], dtype=torch.float32)

    input_dim = X_train.shape[1]
    model = LSTMFeatureExtractor(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    return model


def extract_lstm_features(
    model: LSTMFeatureExtractor,
    X: np.ndarray,
    lookback: int = 20,
) -> pd.DataFrame:
    """Extract LSTM hidden state features for all bars (after lookback warmup).

    Returns DataFrame with columns lstm_h_0 .. lstm_h_{hidden_dim-1}.
    First (lookback-1) rows are NaN (no sequence available yet).
    """
    seqs, _ = build_sequences(X, lookback=lookback)
    X_tensor = torch.tensor(seqs, dtype=torch.float32)

    hidden = model.extract_features(X_tensor)  # (n_seqs, hidden_dim)

    n_total = X.shape[0]
    n_seqs = hidden.shape[0]
    hidden_dim = hidden.shape[1]

    # Pad with NaN for warmup bars
    full = np.full((n_total, hidden_dim), np.nan, dtype=np.float32)
    full[lookback - 1:] = hidden

    cols = [f"lstm_h_{i}" for i in range(hidden_dim)]
    return pd.DataFrame(full, columns=cols)
