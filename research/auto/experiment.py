def configure() -> dict:
    return {'strategy': 'mean_reversion', 'instruments': ['AUD_JPY'], 'timeframe': 'H1', 'vwap_anchor': 46, 'regime_filter': 'conf_donchian_pos_20', 'tier_grid': 'conservative', 'spread_bps': 0.0, 'slippage_bps': 0.0, 'is_bars': 30000, 'oos_bars': 7500, 'description': 'MR AUD_JPY v46 don sp0'}
