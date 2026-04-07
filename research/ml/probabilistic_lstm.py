"""probabilistic_lstm.py -- Phase 1C: Gaussian Output LSTM.

Predicts return distribution (mean + variance) for calibrated probabilities
and principled Kelly position sizing. Fixes Phase 0E's isotonic overfitting
by learning variance end-to-end via GaussianNLLLoss.

Architecture:
    LayerNorm(72) → LSTM(32, 1 layer) → Dropout(0.3) → [mu_head, log_sigma_head]
    Loss: GaussianNLLLoss(mu, target_return, exp(log_sigma))

Probability derivation: P(profit|long) = 1 - Phi(-mu/sigma)
Kelly sizing: f* = mu / sigma^2 (quarter-Kelly for safety)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset

from research.ml.lstm_features import build_sequences


class ProbabilisticLSTM(nn.Module):
    """LSTM with Gaussian output (mean + log-variance)."""

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
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) each of shape (batch,)."""
        x = self.layer_norm(x)
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        return mu, log_sigma


def train_probabilistic_lstm(
    X_train: np.ndarray,
    y_returns: np.ndarray,
    lookback: int = 20,
    hidden_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    patience: int = 7,
) -> ProbabilisticLSTM | None:
    """Train probabilistic LSTM on IS data.

    Args:
        X_train: Feature matrix (n_bars, n_features).
        y_returns: Next-bar returns (n_bars,). NOT binary labels.
    """
    seqs, labels = build_sequences(X_train, y_returns, lookback)
    if labels is None or len(seqs) < 50:
        return None

    # Filter out NaN returns
    valid = np.isfinite(labels)
    seqs = seqs[valid]
    labels = labels[valid]

    n = len(seqs)
    val_start = int(n * 0.8)

    X_tr = torch.tensor(seqs[:val_start], dtype=torch.float32)
    y_tr = torch.tensor(labels[:val_start], dtype=torch.float32)
    X_val = torch.tensor(seqs[val_start:], dtype=torch.float32)
    y_val = torch.tensor(labels[val_start:], dtype=torch.float32)

    if len(X_tr) < 30 or len(X_val) < 10:
        return None

    input_dim = X_train.shape[1]
    model = ProbabilisticLSTM(input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.GaussianNLLLoss(reduction="mean")

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            mu, log_sigma = model(xb)
            variance = torch.exp(2 * log_sigma).clamp(min=1e-6)
            loss = criterion(mu, yb, variance)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            mu_v, ls_v = model(X_val)
            var_v = torch.exp(2 * ls_v).clamp(min=1e-6)
            val_loss = criterion(mu_v, y_val, var_v).item()

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


def predict_probabilistic(
    model: ProbabilisticLSTM,
    X: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict (mu, sigma) for all bars.

    First (lookback-1) bars get mu=0, sigma=1 (uninformative prior).
    Returns (mu_array, sigma_array) each of length n_total.
    """
    seqs, _ = build_sequences(X, lookback=lookback)
    X_tensor = torch.tensor(seqs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        mu, log_sigma = model(X_tensor)
        mu_np = mu.cpu().numpy()
        sigma_np = torch.exp(log_sigma).cpu().numpy()

    n_total = X.shape[0]
    mu_full = np.zeros(n_total, dtype=np.float64)
    sigma_full = np.ones(n_total, dtype=np.float64)
    mu_full[lookback - 1:] = mu_np
    sigma_full[lookback - 1:] = np.clip(sigma_np, 1e-6, None)

    return mu_full, sigma_full


def probabilistic_position(
    mu: np.ndarray,
    sigma: np.ndarray,
    kelly_fraction: float = 0.25,
    min_prob: float = 0.55,
) -> np.ndarray:
    """Convert Gaussian forecast to Kelly-sized positions.

    Args:
        mu: Predicted mean return per bar.
        sigma: Predicted std dev per bar.
        kelly_fraction: Fraction of Kelly (0.25 = quarter-Kelly).
        min_prob: Minimum P(profit) to trade.

    Returns positions in [-1, 1].
    """
    n = len(mu)
    positions = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s = max(sigma[i], 1e-8)
        m = mu[i]

        p_long = 1.0 - norm.cdf(0, loc=m, scale=s)
        p_short = norm.cdf(0, loc=m, scale=s)

        # Kelly: f* = mu / sigma^2
        kelly_long = max(0.0, m / (s ** 2)) * kelly_fraction
        kelly_short = max(0.0, -m / (s ** 2)) * kelly_fraction

        if p_long > min_prob and p_long > p_short:
            positions[i] = min(kelly_long, 1.0)
        elif p_short > min_prob and p_short > p_long:
            positions[i] = -min(kelly_short, 1.0)

    return positions


def compute_brier_score(
    mu: np.ndarray,
    sigma: np.ndarray,
    actual_returns: np.ndarray,
) -> float:
    """Brier score of P(profit|long) predictions. Lower is better (<0.25 = useful)."""
    valid = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(actual_returns)
    mu_v = mu[valid]
    sigma_v = np.clip(sigma[valid], 1e-8, None)
    rets_v = actual_returns[valid]

    if len(mu_v) < 20:
        return 1.0

    p_profit = 1.0 - norm.cdf(0, loc=mu_v, scale=sigma_v)
    actual_profit = (rets_v > 0).astype(float)

    return float(np.mean((p_profit - actual_profit) ** 2))
