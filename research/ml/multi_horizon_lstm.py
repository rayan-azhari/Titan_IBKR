"""multi_horizon_lstm.py -- Phase 1D: Multi-Horizon LSTM.

Shared LSTM backbone predicts 1-bar, 5-bar, and 20-bar returns simultaneously.
Multi-task learning regularizes the backbone, reducing fold-to-fold variance.

Architecture:
    LayerNorm(72) → LSTM(32, 1 layer) → Dropout(0.3) → 3 heads:
        Linear(32→1): 1-bar return  (weight 0.5)
        Linear(32→1): 5-bar return  (weight 0.3)
        Linear(32→1): 20-bar return (weight 0.2)
    Loss = weighted MSE across horizons

Only the 1-bar head drives trading decisions. 5/20-bar heads are regularizers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research.ml.lstm_features import build_sequences

HORIZONS = [1, 5, 20]
HORIZON_WEIGHTS = [0.5, 0.3, 0.2]


class MultiHorizonLSTM(nn.Module):
    """LSTM with multiple return-prediction heads."""

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
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in HORIZONS])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns list of (batch,) tensors, one per horizon."""
        x = self.layer_norm(x)
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return [head(h).squeeze(-1) for head in self.heads]


def build_multi_horizon_targets(
    close: np.ndarray,
) -> list[np.ndarray]:
    """Build forward return targets for each horizon.

    Returns list of arrays, each of length n. NaN where forward return
    is unavailable (end of series).
    """
    n = len(close)
    targets = []
    for h in HORIZONS:
        ret = np.full(n, np.nan, dtype=np.float64)
        for i in range(n - h):
            ret[i] = (close[i + h] - close[i]) / close[i]
        targets.append(ret)
    return targets


def train_multi_horizon_lstm(
    X_train: np.ndarray,
    targets: list[np.ndarray],
    lookback: int = 20,
    hidden_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    patience: int = 7,
) -> MultiHorizonLSTM | None:
    """Train multi-horizon LSTM on IS data.

    Args:
        X_train: Feature matrix (n_bars, n_features).
        targets: List of return arrays per horizon.
    """
    # Build sequences using 1-bar target for alignment
    seqs, _ = build_sequences(X_train, targets[0], lookback)
    if seqs is None or len(seqs) < 50:
        return None

    # Align all targets to sequence indices
    seq_targets = []
    for t in targets:
        seq_targets.append(t[lookback - 1 :])

    # Valid mask: all horizons must have valid targets
    n_seq = len(seqs)
    valid = np.ones(n_seq, dtype=bool)
    for st in seq_targets:
        valid &= np.isfinite(st[:n_seq])

    seqs = seqs[valid]
    seq_targets = [st[:n_seq][valid] for st in seq_targets]

    if len(seqs) < 50:
        return None

    n = len(seqs)
    val_start = int(n * 0.8)

    X_tr = torch.tensor(seqs[:val_start], dtype=torch.float32)
    X_val = torch.tensor(seqs[val_start:], dtype=torch.float32)

    y_tr = [torch.tensor(st[:val_start], dtype=torch.float32) for st in seq_targets]
    y_val = [torch.tensor(st[val_start:], dtype=torch.float32) for st in seq_targets]

    if len(X_tr) < 30 or len(X_val) < 10:
        return None

    input_dim = X_train.shape[1]
    model = MultiHorizonLSTM(input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(X_tr, *y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    weights = torch.tensor(HORIZON_WEIGHTS, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            xb = batch[0]
            ybs = batch[1:]
            preds = model(xb)
            loss = sum(w * nn.functional.mse_loss(p, y) for w, p, y in zip(weights, preds, ybs))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = sum(
                w * nn.functional.mse_loss(p, y) for w, p, y in zip(weights, val_preds, y_val)
            ).item()

        if val_loss < best_val_loss - 1e-6:
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


def predict_multi_horizon(
    model: MultiHorizonLSTM,
    X: np.ndarray,
    lookback: int = 20,
) -> np.ndarray:
    """Predict 1-bar return (from head 0). Returns array of length n_total.

    First (lookback-1) bars get 0.0 (no prediction).
    """
    seqs, _ = build_sequences(X, lookback=lookback)
    X_tensor = torch.tensor(seqs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        pred_1bar = preds[0].cpu().numpy()

    n_total = X.shape[0]
    full = np.zeros(n_total, dtype=np.float64)
    full[lookback - 1 :] = pred_1bar

    return full
