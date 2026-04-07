"""tcn_classifier.py -- Temporal Convolutional Network classifier.

Drop-in replacement for lstm_classifier.py in the stacking ensemble.
TCNs use dilated causal convolutions for longer effective memory (receptive
field = 45 bars vs LSTM's 20-bar lookback) with no vanishing gradient.

Architecture:
    LayerNorm(72) -> 4x TemporalBlock (dilations 1,2,4,8)
    -> last timestep -> Dropout -> Linear(1) -> logits

Interface matches lstm_classifier exactly:
    train_tcn_classifier(X, y, lookback, hidden_dim, ...) -> model
    predict_tcn_classifier(model, X, lookback) -> probabilities
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research.ml.lstm_features import build_sequences


class CausalConv1d(nn.Module):
    """Conv1d with left-padding for causal (no look-ahead) convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Residual block: 2x (CausalConv -> BatchNorm -> ReLU -> Dropout)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(torch.relu(self.bn1(self.conv1(x))))
        out = self.drop(torch.relu(self.bn2(self.conv2(out))))
        return torch.relu(out + self.residual(x))


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network for direction classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        # 4 temporal blocks with exponentially growing dilation
        # Receptive field = kernel_size * sum(dilations) = 3*(1+2+4+8) = 45
        self.blocks = nn.Sequential(
            TemporalBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=8, dropout=dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (batch, lookback, features). Output: (batch,) logits."""
        x = self.layer_norm(x)              # (B, T, F)
        x = x.transpose(1, 2)              # (B, F, T) for Conv1d
        x = self.blocks(x)                 # (B, hidden, T)
        x = x[:, :, -1]                    # take last timestep (B, hidden)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)     # (B,)


def train_tcn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int = 20,
    hidden_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    patience: int = 7,
) -> TCNClassifier | None:
    """Train TCN classifier on IS data with early stopping.

    Interface matches train_lstm_classifier exactly.
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
    model = TCNClassifier(input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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


def predict_tcn_classifier(
    model: TCNClassifier,
    X: np.ndarray,
    lookback: int = 20,
) -> np.ndarray:
    """Predict P(long) for all bars. Returns array of length n_total.

    First (lookback-1) bars get 0.5 (neutral). Matches LSTM interface.
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
