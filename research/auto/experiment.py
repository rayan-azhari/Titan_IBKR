def configure() -> dict:
    return {'strategy': 'mean_reversion', 'instruments': ['AUD_JPY'], 'timeframe': 'H1', 'vwap_anchor': 46, 'regime_filter': 'conf_donchian_pos_20', 'tier_grid': 'conservative', 'spread_bps': 0.5, 'slippage_bps': 0.2, 'is_bars': 28000, 'oos_bars': 8000, 'description': 'MR AUD_JPY v46 sp0.5 is28k oos8k'}
