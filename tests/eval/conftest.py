"""Shared fixtures for the eval test suite."""

import pytest
from dotenv import load_dotenv

load_dotenv()

from app.config import get_llm
from app.state import SentimentReport, MarketQuantReport, SECReport


# ---------------------------------------------------------------------------
# Register the "slow" marker so pytest doesn't emit warnings
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that call external APIs (deselect with '-m \"not slow\"')")


# ---------------------------------------------------------------------------
# Eval LLM instance (shared across tests in a session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_llm():
    """Return a Gemini LLM instance for judge evaluations."""
    return get_llm()


# ---------------------------------------------------------------------------
# Mock report builders
# ---------------------------------------------------------------------------

@pytest.fixture()
def build_sentiment_report():
    """Factory for SentimentReport with sensible defaults."""
    def _build(**overrides):
        defaults = dict(
            ticker="TEST",
            sentiment_score=0.3,
            volume_change_pct=5.0,
            key_narratives=["Earnings beat expectations", "New product launch"],
            raw_evidence=["Source A: positive tone", "Source B: moderate optimism"],
            confidence=0.8,
        )
        defaults.update(overrides)
        return SentimentReport(**defaults)
    return _build


@pytest.fixture()
def build_market_quant_report():
    """Factory for MarketQuantReport with sensible defaults."""
    def _build(**overrides):
        defaults = dict(
            ticker="TEST",
            current_price=150.0,
            price_change_30d_pct=2.5,
            volatility_30d=0.20,
            sharpe_ratio=1.2,
            rsi_14=55.0,
            technical_signal="neutral",
            raw_data_summary="Price stable with moderate volume.",
            confidence=0.85,
        )
        defaults.update(overrides)
        return MarketQuantReport(**defaults)
    return _build


@pytest.fixture()
def build_sec_report():
    """Factory for SECReport with sensible defaults."""
    def _build(**overrides):
        defaults = dict(
            ticker="TEST",
            filing_type="10-K",
            risk_factors=["Market competition", "Regulatory changes"],
            key_financial_metrics={"PE": 25.0, "debt_to_equity": 1.5},
            red_flags=[],
            raw_evidence=["SEC filing 10-K dated 2025-01-15"],
            confidence=0.8,
        )
        defaults.update(overrides)
        return SECReport(**defaults)
    return _build
