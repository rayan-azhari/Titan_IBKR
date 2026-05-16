"""Roll-stitching engine: per-contract bars -> continuous M1/M2 series.

Pre-Reg: ``directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures 2026-05-15.md``.

Roll convention (pre-committed, V3.1):
    - Roll from current front contract to the next 5 business days before
      the current front's ``lastTradeDate``.
    - Back-adjust by ratio: when M1 designation switches from contract A
      to contract B at date ``t_roll``, multiply all prior bars of contract
      A by ``B(t_ref) / A(t_ref)`` where ``t_ref`` is the last calendar
      date on which BOTH A and B have prices. This is the standard
      front-of-curve multiplicative back-adjustment (price-only, no roll
      yield baked in beyond what the ratio already captures).

Causality note (L04 / A1):
    The stitched output is a strictly historical time series. The signal
    layer (D2 carry / B4 TSMOM) still does its own ``.shift(1)`` before
    earning a return.

L40 mitigation:
    The pre-roll buffer keeps the stitching out of the M1 expiry blow-off
    week. Validate by checking that single-bar log returns across roll
    boundaries have the same distribution as non-roll bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContractMeta:
    """Metadata for one futures contract pulled from IBKR."""

    root: str
    local_symbol: str
    con_id: int
    last_trade_date: pd.Timestamp
    parquet: Path
    bars: pd.DataFrame  # daily OHLCV, index = date-only tz-naive timestamps

    @property
    def key(self) -> str:
        return str(self.con_id) if self.con_id > 0 else self.local_symbol


def _parse_yyyymmdd(s: str) -> pd.Timestamp | None:
    if not s:
        return None
    s = str(s).strip()
    try:
        if len(s) >= 8:
            return pd.Timestamp(datetime.strptime(s[:8], "%Y%m%d"))
        if len(s) == 6:
            return pd.Timestamp(datetime.strptime(s + "01", "%Y%m%d"))
    except ValueError:
        return None
    return None


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out.index.name = "timestamp"
    return out.sort_index()


def load_chain(root: str, data_dir: Path | str = "data/ibkr_futures") -> list[ContractMeta]:
    """Load all per-contract parquets for one commodity root.

    Directory layout (from ``scripts/download_ibkr_futures.py``):
        ``{data_dir}/{ROOT}/{LOCAL_SYMBOL}_D.parquet``
        ``{data_dir}/{ROOT}/_chain_meta.csv``  (lastTradeDate per contract)

    If the sidecar CSV is missing, ``last_trade_date`` falls back to the
    last bar of each parquet — less precise but functional for tests.
    """
    root_dir = Path(data_dir) / root
    if not root_dir.is_dir():
        raise FileNotFoundError(f"no chain directory: {root_dir}")

    meta_file = root_dir / "_chain_meta.csv"
    meta_df: pd.DataFrame | None = None
    if meta_file.exists():
        meta_df = pd.read_csv(meta_file)

    chain: list[ContractMeta] = []
    for parq in sorted(root_dir.glob("*_D.parquet")):
        df = _normalize_index(pd.read_parquet(parq))
        if df.empty:
            continue
        local_symbol = parq.name.replace("_D.parquet", "")
        ltd: pd.Timestamp | None = None
        con_id = -1
        if meta_df is not None:
            row = meta_df[meta_df["parquet"] == parq.name]
            if not row.empty:
                ltd = _parse_yyyymmdd(str(row.iloc[0]["lastTradeDate"]))
                try:
                    con_id = int(row.iloc[0]["conId"])
                except (KeyError, ValueError, TypeError):
                    pass
        if ltd is None:
            ltd = df.index[-1]
        chain.append(
            ContractMeta(
                root=root,
                local_symbol=local_symbol,
                con_id=con_id,
                last_trade_date=ltd,
                parquet=parq,
                bars=df,
            )
        )
    chain.sort(key=lambda c: c.last_trade_date)
    return chain


def _roll_date_for(last_trade: pd.Timestamp, buffer_days: int) -> pd.Timestamp:
    """N business days before last_trade. Uses pandas BDay so weekends
    don't count."""
    return last_trade - pd.tseries.offsets.BDay(buffer_days)


def build_continuous(
    chain: list[ContractMeta],
    *,
    contract_offset: int = 0,
    roll_buffer_days: int = 5,
    column: Literal["close", "open", "high", "low"] = "close",
    back_adjust: bool = True,
) -> pd.Series:
    """Build a continuous price series with contract-rolling.

    Parameters
    ----------
    chain : list of ContractMeta
        Per-contract metadata + bars, sorted by ``last_trade_date``.
    contract_offset : int
        0 = M1 (front), 1 = M2 (second-nearest), etc.
    roll_buffer_days : int
        Business days before each contract's ``last_trade_date`` to roll
        out of it. Default 5 (pre-registered).
    column : str
        Which OHLC column to stitch. Default ``"close"``.
    back_adjust : bool
        If ``True`` (default), apply multiplicative ratio back-adjustment
        at each roll so daily log-returns are preserved and cumulative
        wealth is economically meaningful (correct for TSMOM / holding-
        period returns). If ``False``, return the RAW close of whichever
        contract is at ``contract_offset`` on each date — correct for
        cross-sectional carry signals like ``log(M1_raw/M2_raw)`` where
        independent back-adjustment factors would distort the basis.

    Returns
    -------
    pd.Series
        Continuous price series indexed by date.
    """
    if not chain:
        return pd.Series(dtype=float)
    if contract_offset < 0:
        raise ValueError(f"contract_offset must be >=0, got {contract_offset}")
    if contract_offset >= len(chain):
        return pd.Series(dtype=float)

    chain = sorted(chain, key=lambda c: c.last_trade_date)
    roll_dates = {c.key: _roll_date_for(c.last_trade_date, roll_buffer_days) for c in chain}
    by_key = {c.key: c for c in chain}

    # Pre-sort chain into a numpy view we can advance through.
    sorted_keys = [c.key for c in chain]
    sorted_roll = np.array(
        [roll_dates[k].to_datetime64() for k in sorted_keys], dtype="datetime64[ns]"
    )

    # Universe of dates.
    all_dates = pd.DatetimeIndex(sorted(set().union(*(set(c.bars.index) for c in chain))))
    if len(all_dates) == 0:
        return pd.Series(dtype=float)

    # For each date, find first contract whose roll-date >= t. That is M1.
    # Then advance through subsequent ones for M2, etc.
    rolls64 = sorted_roll  # alias
    dates64 = all_dates.values.astype("datetime64[ns]")

    # For each date, np.searchsorted finds where to insert t to keep the
    # roll array sorted; that's the index of the first roll-date >= t.
    first_active = np.searchsorted(rolls64, dates64, side="left")

    segments_date: list[pd.Timestamp] = []
    segments_price: list[float] = []
    segments_src: list[str] = []
    for i, t in enumerate(all_dates):
        # Try contracts at positions [first_active[i], first_active[i]+1, ...]
        # until we find one with a price on t. Pick the (offset+1)-th
        # such contract.
        cursor = int(first_active[i])
        matched = 0
        while cursor < len(sorted_keys):
            k = sorted_keys[cursor]
            c = by_key[k]
            if t in c.bars.index:
                if matched == contract_offset:
                    px = float(c.bars.at[t, column])
                    if np.isfinite(px):
                        segments_date.append(t)
                        segments_price.append(px)
                        segments_src.append(k)
                    break
                matched += 1
            cursor += 1

    if not segments_date:
        return pd.Series(dtype=float)

    price_array = np.asarray(segments_price, dtype=float)
    src_array = np.asarray(segments_src, dtype=object)

    # Apply ratio back-adjustment at each transition.
    transitions: list[tuple[int, str, str]] = []
    for i in range(1, len(src_array)):
        if src_array[i] != src_array[i - 1]:
            transitions.append((i, src_array[i - 1], src_array[i]))

    adj = np.ones(len(price_array), dtype=float)
    if back_adjust:
        # Walk transitions in reverse so each ratio applies once to the
        # contiguous-prior block.
        for i, old_key, new_key in reversed(transitions):
            old_c = by_key[old_key]
            new_c = by_key[new_key]
            common = old_c.bars.index.intersection(new_c.bars.index)
            if len(common) == 0:
                continue
            ref_date = common.max()
            try:
                old_px = float(old_c.bars.at[ref_date, column])
                new_px = float(new_c.bars.at[ref_date, column])
            except KeyError:
                continue
            if not (np.isfinite(old_px) and np.isfinite(new_px)) or old_px <= 0:
                continue
            ratio = new_px / old_px
            adj[:i] *= ratio

    suffix = "" if back_adjust else "_raw"
    out = pd.Series(
        price_array * adj,
        index=pd.DatetimeIndex(segments_date, name="timestamp"),
        name=f"M{contract_offset + 1}{suffix}",
    )
    return out


def stitch_root(
    root: str,
    *,
    data_dir: Path | str = "data/ibkr_futures",
    out_dir: Path | str = "data",
    roll_buffer_days: int = 5,
    save: bool = True,
) -> dict[str, pd.Series]:
    """Stitch M1 and M2 for one commodity root, emitting both back-adjusted
    and raw variants.

    Returns dict with keys ``"M1"``, ``"M2"``, ``"M1_raw"``, ``"M2_raw"``
    mapping to continuous Series. The first two are back-adjusted (correct
    for holding-period returns / TSMOM). The ``_raw`` variants store the
    actual close of the M1 / M2 contract on each date (correct for
    cross-sectional carry signals).

    If ``save`` is True, writes parquets:
        ``{out_dir}/{root}_M1_stitched_D.parquet``       (back-adjusted)
        ``{out_dir}/{root}_M2_stitched_D.parquet``       (back-adjusted)
        ``{out_dir}/{root}_M1_raw_stitched_D.parquet``   (raw close)
        ``{out_dir}/{root}_M2_raw_stitched_D.parquet``   (raw close)
    """
    chain = load_chain(root, data_dir=data_dir)
    m1 = build_continuous(chain, contract_offset=0, roll_buffer_days=roll_buffer_days)
    m2 = build_continuous(chain, contract_offset=1, roll_buffer_days=roll_buffer_days)
    m1_raw = build_continuous(
        chain, contract_offset=0, roll_buffer_days=roll_buffer_days, back_adjust=False
    )
    m2_raw = build_continuous(
        chain, contract_offset=1, roll_buffer_days=roll_buffer_days, back_adjust=False
    )
    result = {"M1": m1, "M2": m2, "M1_raw": m1_raw, "M2_raw": m2_raw}
    if save and not m1.empty:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for tag, series in result.items():
            if series.empty:
                continue
            series.to_frame(name="close").to_parquet(out_path / f"{root}_{tag}_stitched_D.parquet")
    return result
