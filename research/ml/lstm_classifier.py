"""lstm_classifier.py -- Phase 1A: LSTM End-to-End Classifier.

Replaces XGBoost entirely with a lightweight LSTM that predicts direction
from a lookback window of features. Avoids the Phase 0C failure mode
(feature stuffing) by using LSTM as the final model, not a feature extractor.

Architecture (intentionally small to prevent overfit on ~400 IS sequences):
    LayerNorm(72) → LSTM(32 hidden, 1 layer) → Dropout(0.3) → Linear(1)
    Loss: BCEWithLogitsLoss
    Optimizer: AdamW (weight_decay=1e-3) + OneCycleLR + grad clipping

Key differences from Phase 0C's LSTMFeatureExtractor:
    - End-to-end: no downstream XGBoost
    - Smaller: 32 hidden, 1 layer (~5K params vs ~20K)
    - LayerNorm on input (features have wildly different scales)
    - AdamW with weight decay (stronger regularization)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research.ml.lstm_features import build_sequences


class LSTMClassifier(nn.Module):
    """Lightweight LSTM for end-to-end direction classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (batch,)."""
        x = self.layer_norm(x)
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])  # (batch, hidden)
        return self.head(h).squeeze(-1)


def train_lstm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int = 20,
    hidden_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    patience: int = 7,
) -> LSTMClassifier | None:
    """Train LSTM classifier on IS data with early stopping.

    Returns None if insufficient data.
    """
    seqs, labels = build_sequences(X_train, y_train, lookback)
    if labels is None or len(seqs) < 50:
        return None

    n = len(seqs)
    val_start = int(n * 0.8)

    X_tr = torch.tensor(seqs[:val_start], dtype=torch.float32)
    y_tr = torch.tensor(labels[:val_start], dtype=torch.float32)
    X_val = torch.tensor(seqs[val_start:], dtype=torch.float32)
    y_val = torch.tensor(labels[val_start:], dtype=torch.float32)

    if len(X_tr) < 30 or len(X_val) < 10:
        return None

    input_dim = X_train.shape[1]
    model = LSTMClassifier(input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weighting
    pos_count = float(y_tr.sum())
    neg_count = float(len(y_tr) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()

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
    model.eval()
    return model


def predict_lstm_classifier(
    model: LSTMClassifier,
    X: np.ndarray,
    lookback: int = 20,
) -> np.ndarray:
    """Predict P(long) for all bars. Returns array of length n_total.

    First (lookback-1) bars get 0.5 (neutral).
    """
    seqs, _ = build_sequences(X, lookback=lookback)
    X_tensor = torch.tensor(seqs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        proba = torch.sigmoid(logits).cpu().numpy()

    n_total = X.shape[0]
    full = np.full(n_total, 0.5, dtype=np.float64)
    full[lookback - 1:] = proba

    return full
