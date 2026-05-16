"""GEM noise-robust redesign audit (J4).

Pre-registered in directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md.

Tests three mitigations (A=EWMA, B=position-cap, C=quantile-target) against
the C12 baseline to find which one restores the noise axis to `best` while
preserving the other 4 statistical axes.
"""
