"""
Truth Checker Effectiveness Eval

Tests whether the Truth Checker correctly detects conflicts and consistency
using carefully constructed mock report data. The inputs are mock Pydantic
reports; the truth_checker_node itself calls the real Gemini LLM.
"""

import pytest
from app.agents.truth_checker import truth_checker_node
from app.state import SentimentReport, MarketQuantReport, SECReport


# ---------------------------------------------------------------------------
# Helpers — build mock reports with specific characteristics
# ---------------------------------------------------------------------------

def _bullish_sentiment(**kw):
    defaults = dict(
        ticker="TEST",
        sentiment_score=0.9,
        volume_change_pct=40.0,
        key_narratives=["Massive rally expected", "Institutional buying frenzy"],
        raw_evidence=["Tweet: $TEST to the moon!", "Analyst upgrades across the board"],
        confidence=0.9,
    )
    defaults.update(kw)
    return SentimentReport(**defaults)


def _bearish_quant(**kw):
    defaults = dict(
        ticker="TEST",
        current_price=95.0,
        price_change_30d_pct=-15.0,
        volatility_30d=0.55,
        sharpe_ratio=-1.2,
        rsi_14=82.0,
        technical_signal="bearish",
        raw_data_summary="Price dropped 15% in 30 days with RSI overbought at 82. Sharpe ratio deeply negative.",
        confidence=0.85,
    )
    defaults.update(kw)
    return MarketQuantReport(**defaults)


def _debt_heavy_sec(**kw):
    defaults = dict(
        ticker="TEST",
        filing_type="10-K",
        risk_factors=[
            "Significant long-term debt obligations",
            "Debt covenants may restrict operations",
            "Interest rate risk on variable-rate debt",
        ],
        key_financial_metrics={"debt_to_equity": 4.5, "interest_coverage": 1.2},
        red_flags=["Going concern language in auditor opinion"],
        raw_evidence=["10-K filing 2025: total debt $12B vs equity $2.7B"],
        confidence=0.75,
    )
    defaults.update(kw)
    return SECReport(**defaults)


def _mild_positive_sentiment(**kw):
    defaults = dict(
        ticker="TEST",
        sentiment_score=0.25,
        volume_change_pct=3.0,
        key_narratives=["Steady growth expectations", "Stable dividends"],
        raw_evidence=["News: consistent quarterly performance"],
        confidence=0.8,
    )
    defaults.update(kw)
    return SentimentReport(**defaults)


def _stable_quant(**kw):
    defaults = dict(
        ticker="TEST",
        current_price=150.0,
        price_change_30d_pct=1.5,
        volatility_30d=0.18,
        sharpe_ratio=1.1,
        rsi_14=52.0,
        technical_signal="neutral",
        raw_data_summary="Price stable with low volatility and healthy Sharpe ratio.",
        confidence=0.88,
    )
    defaults.update(kw)
    return MarketQuantReport(**defaults)


def _clean_sec(**kw):
    defaults = dict(
        ticker="TEST",
        filing_type="10-K",
        risk_factors=["General market risk", "Competition"],
        key_financial_metrics={"PE": 22.0, "debt_to_equity": 0.5},
        red_flags=[],
        raw_evidence=["10-K filed 2025-02-15, clean auditor opinion"],
        confidence=0.85,
    )
    defaults.update(kw)
    return SECReport(**defaults)


def _build_state(sentiment=None, quant=None, sec=None, iteration=0):
    return {
        "iteration_count": iteration,
        "sentiment_report": sentiment,
        "sec_report": sec,
        "market_quant_report": quant,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Case A: Obvious conflict — bullish sentiment vs bearish technicals
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_obvious_conflict_detected():
    """Extremely bullish sentiment (+0.9) contradicts bearish technicals (RSI>80, -15% price).
    Truth Checker should detect at least 1 conflict with severity >= medium."""

    state = _build_state(
        sentiment=_bullish_sentiment(),
        quant=_bearish_quant(),
        sec=_clean_sec(),
    )

    result = truth_checker_node(state)
    report = result["truth_check_report"]

    print(f"\n[Case A] Conflicts: {len(report.conflicts)}")
    print(f"[Case A] Consistency: {report.overall_consistency}")
    print(f"[Case A] Recommendation: {report.recommendation}")
    for c in report.conflicts:
        print(f"  - {c.severity}: {c.source_a} vs {c.source_b} — {c.explanation[:80]}")

    assert len(report.conflicts) >= 1, "Should detect at least 1 conflict between bullish sentiment and bearish technicals"

    severities = [c.severity for c in report.conflicts]
    assert any(
        s in ("medium", "high", "critical") for s in severities
    ), f"At least one conflict should be medium+ severity, got {severities}"


# ---------------------------------------------------------------------------
# Case B: Subtle conflict — debt risk vs high-confidence bullish sentiment
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_subtle_conflict_detected():
    """SEC reports major debt risk, but sentiment is very bullish with no debt mention.
    Truth Checker should flag at least a low-severity conflict."""

    state = _build_state(
        sentiment=_bullish_sentiment(
            key_narratives=["Revenue growth acceleration", "Market share gains"],
            raw_evidence=["Analyst: strong revenue outlook", "Tweet: best quarter ever"],
        ),
        quant=_stable_quant(),
        sec=_debt_heavy_sec(),
    )

    result = truth_checker_node(state)
    report = result["truth_check_report"]

    print(f"\n[Case B] Conflicts: {len(report.conflicts)}")
    print(f"[Case B] Consistency: {report.overall_consistency}")
    print(f"[Case B] Recommendation: {report.recommendation}")
    for c in report.conflicts:
        print(f"  - {c.severity}: {c.source_a} vs {c.source_b} — {c.explanation[:80]}")

    assert len(report.conflicts) >= 1, "Should flag debt risk vs bullish sentiment mismatch"


# ---------------------------------------------------------------------------
# Case C: Consistent reports — all three align
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_consistent_reports():
    """All three reports align: mildly positive sentiment, stable technicals, clean SEC.
    Truth Checker should return high consistency (>= 0.7)."""

    state = _build_state(
        sentiment=_mild_positive_sentiment(),
        quant=_stable_quant(),
        sec=_clean_sec(),
    )

    result = truth_checker_node(state)
    report = result["truth_check_report"]

    print(f"\n[Case C] Conflicts: {len(report.conflicts)}")
    print(f"[Case C] Consistency: {report.overall_consistency}")
    print(f"[Case C] Recommendation: {report.recommendation}")

    assert report.overall_consistency >= 0.7, (
        f"Consistent reports should yield consistency >= 0.7, got {report.overall_consistency}"
    )
    # Recommendation should be consistent or minor_conflicts (not major or needs_more_data in a harsh sense)
    # Note: round 0 prompt may lean toward needs_more_data, so we accept that too
    assert report.recommendation in ("consistent", "minor_conflicts", "needs_more_data"), (
        f"Expected consistent/minor_conflicts/needs_more_data, got {report.recommendation}"
    )


# ---------------------------------------------------------------------------
# Case D: Missing data — only 1 of 3 reports available
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_missing_data_handled():
    """Only sentiment report available (quant and SEC missing).
    Truth Checker should NOT return 'consistent'."""

    state = _build_state(
        sentiment=_mild_positive_sentiment(),
        quant=None,
        sec=None,
    )

    result = truth_checker_node(state)
    report = result["truth_check_report"]

    print(f"\n[Case D] Conflicts: {len(report.conflicts)}")
    print(f"[Case D] Consistency: {report.overall_consistency}")
    print(f"[Case D] Recommendation: {report.recommendation}")

    assert report.recommendation != "consistent", (
        "With 2 of 3 reports missing, should NOT be 'consistent'"
    )
