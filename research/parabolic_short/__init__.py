"""Parabolic Short strategy research (David Hanlin Module 2 mechanization).

Tests whether mean-reversion edge exists after extreme parabolic gap-ups in
liquid US equities, applying the Hanlin-style daily-chart filter set:
  3+ consecutive green days, volume blowoff, gap up, extension from 10dMA,
  reversal confirmation (red close on the gap day).

Entry on next-day open after a red parabolic-day close, stop at parabolic
high, target 10dSMA reclaim. Daily-only: no intraday data required.
"""
