"""experiment.py -- Agent-modifiable strategy configuration.

THIS IS THE ONLY FILE THE AUTONOMOUS AGENT SHOULD MODIFY.

The configure() function returns a dict with all strategy parameters.
The evaluate.py harness imports this, runs WFO, and reports the score.

Current baseline: XGBoost+LSTM stacking on QQQ+SPY (best known config).
"""


def configure() -> dict:
    """Return the full strategy configuration.

    The agent modifies this function to test hypotheses.
    Each modification = one experiment. Keep changes small and focused.
    """
    return {
        # ─── Metadata ────────────────────────────────────────────────
        "description": "Exp38: IWB only oos3",

        # ─── Strategy Selection ──────────────────────────────────────
        # Options: "xgboost", "stacking", "lstm_e2e"
        "strategy": "stacking",

        # ─── Instruments ─────────────────────────────────────────────
        # Must exist in data/ as {INSTRUMENT}_{TIMEFRAME}.parquet
        "instruments": ["IWB"],
        "timeframe": "D",

        # ─── XGBoost Parameters ──────────────────────────────────────
        "xgb_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "random_state": 42,
            "verbosity": 0,
        },

        # ─── LSTM Parameters (used by stacking and lstm_e2e) ────────
        "lstm_hidden": 32,
        "lookback": 20,
        "lstm_epochs": 30,

        # ─── Stacking-Specific ───────────────────────────────────────
        "n_nested_folds": 3,

        # ─── Label Engineering ───────────────────────────────────────
        # Regime+pullback labeler parameters to sweep per fold.
        # More entries = more options per fold (best is auto-selected).
        "label_params": [
            {
                "rsi_oversold": 45,
                "rsi_overbought": 55,
                "confirm_bars": 10,
                "confirm_pct": 0.005,
            },
            {
                "rsi_oversold": 50,
                "rsi_overbought": 50,
                "confirm_bars": 10,
                "confirm_pct": 0.003,
            },
            {
                "rsi_oversold": 48,
                "rsi_overbought": 52,
                "confirm_bars": 10,
                "confirm_pct": 0.005,
            },
        ],

        # ─── Position Sizing ─────────────────────────────────────────
        "signal_threshold": 0.6,  # P(long) > this = go long
        "cost_bps": 2.0,          # Transaction cost in basis points

        # ─── WFO Configuration ───────────────────────────────────────
        "is_years": 2,     # In-sample window (years)
        "oos_months": 3,   # Out-of-sample window (months)
    }
