"""
Test 1: Market Quant agent — API failure graceful handling + RSI range validation

Tests that when yfinance fails, market_quant_node returns an error dict
instead of crashing the entire graph.
Also validates RSI-14 calculation stays within 0-100 range.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from app.agents.market_quant import market_quant_node, _fetch_stock_data


# ---------------------------------------------------------------------------
# Test 1a: API failure → node returns {"errors": [...]} instead of crashing
# ---------------------------------------------------------------------------

@patch("app.agents.market_quant._fetch_stock_data")
def test_market_quant_returns_error_on_api_failure(mock_fetch):
    """When yfinance throws, the node should return an errors list, not crash."""
    mock_fetch.side_effect = Exception("API timeout")

    fake_state = {
        "ticker": "AAPL",
        "market_quant_report": None,
    }

    result = market_quant_node(fake_state)

    assert "errors" in result, "Node should return errors key on failure"
    assert len(result["errors"]) == 1
    assert "Market Quant failed" in result["errors"][0]
    assert "API timeout" in result["errors"][0]


# ---------------------------------------------------------------------------
# Test 1b: Skip on round 2 — if report already exists, return empty dict
# ---------------------------------------------------------------------------

def test_market_quant_reuses_existing_report():
    """On round 2, if a report already exists, node should skip and return {}."""
    from app.state import MarketQuantReport

    existing_report = MarketQuantReport(
        ticker="AAPL",
        current_price=190.0,
        price_change_30d_pct=2.5,
        volatility_30d=0.25,
        sharpe_ratio=1.2,
        rsi_14=55.0,
        technical_signal="neutral",
        raw_data_summary="Test summary",
        confidence=0.85,
    )

    fake_state = {
        "ticker": "AAPL",
        "market_quant_report": existing_report,
    }

    result = market_quant_node(fake_state)

    # Should return empty dict (skip), no API call made
    assert result == {}


# ---------------------------------------------------------------------------
# Test 1c: RSI-14 value is always within [0, 100]
# ---------------------------------------------------------------------------

def test_rsi_calculation_within_valid_range():
    """RSI-14 should always be between 0 and 100, regardless of price data."""

    # Build fake 60-day price history (enough for RSI-14)
    np.random.seed(42)
    dates = pd.bdate_range(start="2025-01-01", periods=60)
    # Simulate a volatile stock
    price_changes = np.random.normal(0, 0.03, 60)
    prices = 100 * np.cumprod(1 + price_changes)
    volumes = np.random.randint(1_000_000, 10_000_000, 60)

    fake_hist = pd.DataFrame(
        {"Close": prices, "Volume": volumes},
        index=dates,
    )

    with patch("app.agents.market_quant.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = fake_hist
        mock_ticker.return_value = mock_instance

        data = _fetch_stock_data("FAKE")

    assert 0 <= data["rsi_14"] <= 100, f"RSI {data['rsi_14']} is out of [0, 100]"
    assert data["current_price"] > 0
    assert isinstance(data["sharpe_ratio"], float)


def test_rsi_stays_valid_on_downtrend():
    """RSI should stay valid even with a consistent downtrend (edge case)."""
    dates = pd.bdate_range(start="2025-01-01", periods=60)
    # Pure downtrend: price drops every day
    prices = [100 - i * 0.5 for i in range(60)]
    volumes = [5_000_000] * 60

    fake_hist = pd.DataFrame(
        {"Close": prices, "Volume": volumes},
        index=dates,
    )

    with patch("app.agents.market_quant.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = fake_hist
        mock_ticker.return_value = mock_instance

        data = _fetch_stock_data("SINK")

    assert 0 <= data["rsi_14"] <= 100, f"RSI {data['rsi_14']} out of range on downtrend"
    # In a pure downtrend, RSI should be low
    assert data["rsi_14"] < 40, f"RSI {data['rsi_14']} unexpectedly high for pure downtrend"


def test_fetch_stock_data_raises_on_empty_history():
    """When yfinance returns empty DataFrame, _fetch_stock_data should raise."""
    with patch("app.agents.market_quant.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        with pytest.raises(ValueError, match="Could not retrieve stock data"):
            _fetch_stock_data("INVALID")
