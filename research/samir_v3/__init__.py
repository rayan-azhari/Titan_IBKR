"""Samir-Stack V3 — fresh re-derivation around VIX-HMM risk classification.

See ``directives/Samir V3 — VIX-HMM Strategy Design 2026-05-13.md`` for the
architectural rationale.

V3 builds the strategy bottom-up:
  Layer 1: VIX-HMM regime classifier (this module)
  Layer 2: Pure equity, binary deploy/cash via MES futures
  Layer 3: Validation vs baselines
  (Layers 4+: momentum, bonds, capitulation — only if earlier layers pass)
"""
