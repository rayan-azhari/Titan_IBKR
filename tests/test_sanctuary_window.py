"""Tests for the 12-month sanctuary window in ``research.auto.evaluate``.

The sanctuary window is the most recent 12 months of data. The
autoresearch agent cycles through thousands of experiments and will
eventually overfit to whatever price action it can see — so the last
year is strictly reserved for the human operator's final release-gate
validation run. These tests lock in:

* that the guard actually trims bars in the sanctuary window;
* that the ``--include-sanctuary`` CLI flag and the
  ``TITAN_INCLUDE_SANCTUARY`` env var both disable the guard;
* that the default start-of-sanctuary is ~365 days before now.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest


def _sample_df(days: int = 3 * 365) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    idx = pd.date_range(end=end, periods=days, freq="D", tz="UTC")
    return pd.DataFrame({"close": range(len(idx))}, index=idx)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Ensure each test starts with a clean sanctuary env / argv state."""
    monkeypatch.delenv("TITAN_INCLUDE_SANCTUARY", raising=False)
    # Strip any leftover --include-sanctuary from earlier tests.
    monkeypatch.setattr(sys, "argv", [a for a in sys.argv if a != "--include-sanctuary"])


def test_sanctuary_trims_last_year_by_default():
    from research.auto.evaluate import _enforce_sanctuary, _sanctuary_start

    df = _sample_df()
    trimmed = _enforce_sanctuary(df, source="test")
    assert len(trimmed) < len(df), "sanctuary guard must drop recent bars by default"
    # All kept bars must be strictly before the sanctuary start.
    assert trimmed.index.max() < _sanctuary_start()


def test_sanctuary_disabled_by_flag(monkeypatch):
    from research.auto import evaluate

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--include-sanctuary"])
    df = _sample_df()
    out = evaluate._enforce_sanctuary(df, source="test")
    assert len(out) == len(df)


def test_sanctuary_disabled_by_env(monkeypatch):
    from research.auto import evaluate

    monkeypatch.setenv("TITAN_INCLUDE_SANCTUARY", "1")
    df = _sample_df()
    out = evaluate._enforce_sanctuary(df, source="test")
    assert len(out) == len(df)


def test_sanctuary_start_is_one_year_before_end():
    from research.auto.evaluate import SANCTUARY_DAYS, _sanctuary_start

    end = datetime(2026, 4, 21, tzinfo=timezone.utc)
    start = _sanctuary_start(end=end, days=SANCTUARY_DAYS)
    expected = pd.Timestamp(end - timedelta(days=SANCTUARY_DAYS)).tz_convert("UTC")
    assert start == expected


def test_sanctuary_noop_on_empty_df():
    from research.auto.evaluate import _enforce_sanctuary

    df = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([], tz="UTC"))
    out = _enforce_sanctuary(df, source="empty")
    assert len(out) == 0


def test_sanctuary_noop_on_non_datetime_index():
    from research.auto.evaluate import _enforce_sanctuary

    df = pd.DataFrame({"close": [1.0, 2.0]})
    out = _enforce_sanctuary(df, source="non-datetime")
    assert len(out) == len(df)


def test_sanctuary_handles_tz_naive_index():
    from research.auto.evaluate import _enforce_sanctuary

    end = datetime.now()
    idx = pd.date_range(end=end, periods=800, freq="D")  # tz-naive
    df = pd.DataFrame({"close": range(len(idx))}, index=idx)
    # Should not raise; should still trim some bars.
    out = _enforce_sanctuary(df, source="tz-naive")
    assert 0 < len(out) < len(df)


def test_loaders_respect_sanctuary(monkeypatch):
    """End-to-end: ``_load_daily`` trims the sanctuary window.

    Uses a monkeypatched ``pd.read_parquet`` rather than real file I/O so
    the test doesn't depend on pyarrow global-state behavior when many
    tests are collected together.
    """
    from research.auto import evaluate

    sym = "SANCT_TEST"
    end = datetime.now(timezone.utc)
    idx = pd.date_range(end=end, periods=800, freq="D", tz="UTC")
    fake_df = pd.DataFrame({"close": range(len(idx))}, index=idx)

    # Pretend the parquet exists.
    monkeypatch.setattr(
        "pathlib.Path.exists",
        lambda self: str(self).endswith(f"{sym}_D.parquet"),
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: fake_df.copy())

    df = evaluate._load_daily(sym)
    assert df.index.max() < evaluate._sanctuary_start()


def test_loaders_release_gate_includes_sanctuary(monkeypatch):
    """With the override active, ``_load_daily`` keeps every bar."""
    from research.auto import evaluate

    sym = "RELEASE_GATE"
    end = datetime.now(timezone.utc)
    idx = pd.date_range(end=end, periods=800, freq="D", tz="UTC")
    fake_df = pd.DataFrame({"close": range(len(idx))}, index=idx)

    monkeypatch.setattr(
        "pathlib.Path.exists",
        lambda self: str(self).endswith(f"{sym}_D.parquet"),
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: fake_df.copy())
    monkeypatch.setenv("TITAN_INCLUDE_SANCTUARY", "1")

    df = evaluate._load_daily(sym)
    assert len(df) == len(idx)


def test_sanctuary_env_var_respects_falsy_values(monkeypatch):
    from research.auto import evaluate

    monkeypatch.setenv("TITAN_INCLUDE_SANCTUARY", "0")
    df = _sample_df()
    trimmed = evaluate._enforce_sanctuary(df, source="falsy")
    # 0 / empty should NOT disable the guard — only 1/true/TRUE.
    assert len(trimmed) < len(df)

    # Also check that an unrelated value leaves the guard active.
    monkeypatch.setenv("TITAN_INCLUDE_SANCTUARY", "maybe")
    trimmed = evaluate._enforce_sanctuary(df, source="maybe")
    assert len(trimmed) < len(df)


_ = os  # silence import-lint on os (used by env-based tests via monkeypatch)
