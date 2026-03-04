"""
Test 2: Truth Checker — loop-back routing logic

Tests the route_after_truth_check function to verify when the graph
loops back to Supervisor vs. proceeds to Report Generator.
Also tests the no-reports edge case.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.state import (
    TruthCheckReport,
    Conflict,
    SentimentReport,
    MarketQuantReport,
    SECReport,
)
from app.graph import route_after_truth_check


# ---------------------------------------------------------------------------
# Helper: build a TruthCheckReport with a given recommendation
# ---------------------------------------------------------------------------

def _make_truth_report(recommendation: str, conflicts: list[Conflict] | None = None):
    return TruthCheckReport(
        conflicts=conflicts or [],
        overall_consistency=0.7,
        recommendation=recommendation,
        summary="Test summary",
    )


# ---------------------------------------------------------------------------
# Test 2a: needs_more_data + under max iterations → loop back to supervisor
# ---------------------------------------------------------------------------

def test_route_loops_back_on_needs_more_data_round1():
    """Round 1 with needs_more_data should route back to supervisor."""
    state = {
        "truth_check_report": _make_truth_report("needs_more_data"),
        "iteration_count": 1,   # Just finished round 1
        "max_iterations": 2,
    }

    result = route_after_truth_check(state)
    assert result == "supervisor", "Should loop back to supervisor for re-investigation"


# ---------------------------------------------------------------------------
# Test 2b: needs_more_data but already at max iterations → go to report
# ---------------------------------------------------------------------------

def test_route_goes_to_report_when_at_max_iterations():
    """Even with needs_more_data, if at max iterations, stop looping."""
    state = {
        "truth_check_report": _make_truth_report("needs_more_data"),
        "iteration_count": 2,   # Already hit max
        "max_iterations": 2,
    }

    result = route_after_truth_check(state)
    assert result == "report_generator", "Should NOT loop back when at max iterations"


# ---------------------------------------------------------------------------
# Test 2c: consistent → go directly to report
# ---------------------------------------------------------------------------

def test_route_goes_to_report_when_consistent():
    """When reports are consistent, go straight to report generator."""
    state = {
        "truth_check_report": _make_truth_report("consistent"),
        "iteration_count": 1,
        "max_iterations": 2,
    }

    result = route_after_truth_check(state)
    assert result == "report_generator"


# ---------------------------------------------------------------------------
# Test 2d: minor_conflicts → go to report (not loop)
# ---------------------------------------------------------------------------

def test_route_goes_to_report_on_minor_conflicts():
    """minor_conflicts should proceed to report, not trigger re-investigation."""
    conflict = Conflict(
        source_a="Sentiment Scout",
        claim_a="Bullish sentiment",
        source_b="Market Quant",
        claim_b="Neutral technicals",
        severity="low",
        explanation="Tone difference, not factual.",
    )

    state = {
        "truth_check_report": _make_truth_report("minor_conflicts", [conflict]),
        "iteration_count": 1,
        "max_iterations": 2,
    }

    result = route_after_truth_check(state)
    assert result == "report_generator"


# ---------------------------------------------------------------------------
# Test 2e: no truth_check_report at all → fallback to report generator
# ---------------------------------------------------------------------------

def test_route_fallback_when_no_report():
    """If truth_check_report is somehow None, should still go to report."""
    state = {
        "truth_check_report": None,
        "iteration_count": 1,
        "max_iterations": 2,
    }

    result = route_after_truth_check(state)
    assert result == "report_generator"


# ---------------------------------------------------------------------------
# Test 2f: truth_checker_node returns needs_more_data when no reports exist
# ---------------------------------------------------------------------------

def test_truth_checker_node_handles_no_reports():
    """When all 3 agent reports are None, truth checker should say needs_more_data."""
    from app.agents.truth_checker import truth_checker_node

    fake_state = {
        "iteration_count": 0,
        "sentiment_report": None,
        "sec_report": None,
        "market_quant_report": None,
    }

    result = truth_checker_node(fake_state)

    report = result["truth_check_report"]
    assert report.recommendation == "needs_more_data"
    assert report.overall_consistency == 0.0
    assert result["iteration_count"] == 1  # Should increment
