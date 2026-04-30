"""Multi-timeframe Market Structure Shift (MSS) trend strategy research.

External-source idea: Daily-trend / 15M-MSS-entry FX trend-following, claimed
957% / 72% WR / 50% DD over 10 years across 5 CME FX pairs. The published
result is a max-of-8,700-grid order statistic, so the prior on real edge is
much lower than the headline. This module tests the exact spec under honest
IS/OOS discipline with bootstrap CIs, no in-sample tuning.
"""
