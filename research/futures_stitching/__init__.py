"""Futures roll-stitching engine.

Pre-registered in ``directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures
2026-05-15.md``.

Given per-contract daily bars from IBKR (one parquet per expiry under
``data/ibkr_futures/{ROOT}/{LOCAL_SYMBOL}_D.parquet``), produces two
continuous-price series per commodity root:

    - M1 (front-month, back-adjusted by ratio at each roll).
    - M2 (second-month, same convention).

The roll convention is pre-committed in the directive: roll 5 business
days before each contract's ``lastTradeDate``. Back-adjustment is the
ratio method, applied retroactively at each roll boundary.
"""

from research.futures_stitching.stitch import (
    ContractMeta,
    build_continuous,
    load_chain,
    stitch_root,
)

__all__ = ["ContractMeta", "build_continuous", "load_chain", "stitch_root"]
