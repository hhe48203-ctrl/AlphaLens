"""
Test 3: Supervisor — ticker resolution failure → graph exits gracefully

Tests that when LLM cannot resolve a ticker (or LLM call itself fails),
the supervisor returns status="completed" so the graph exits via
route_after_supervisor → END, instead of crashing.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.agents.supervisor import supervisor_node, TickerResolution
from app.graph import route_after_supervisor


# ---------------------------------------------------------------------------
# Test 3a: LLM says resolved=False → status="completed"
# ---------------------------------------------------------------------------

@patch("app.agents.supervisor._resolve_ticker")
def test_supervisor_exits_when_ticker_unresolvable(mock_resolve):
    """If LLM cannot resolve the ticker, supervisor should set status=completed."""
    mock_resolve.return_value = TickerResolution(
        ticker="",
        company_name="",
        reasoning="The input 'sdfkjhsdf' does not refer to any known stock.",
        resolved=False,
    )

    fake_state = {
        "user_query": "sdfkjhsdf",
        "iteration_count": 0,
    }

    result = supervisor_node(fake_state)

    assert result.get("status") == "completed", "Should exit graph on unresolvable ticker"
    assert "ticker" not in result, "Should NOT set ticker when unresolved"


# ---------------------------------------------------------------------------
# Test 3b: route_after_supervisor routes to END when status=completed
# ---------------------------------------------------------------------------

def test_route_after_supervisor_goes_to_end():
    """When supervisor sets status=completed, routing should go to __end__."""
    state = {"status": "completed"}

    result = route_after_supervisor(state)

    assert result == ["__end__"], "Should route to END when status is completed"


# ---------------------------------------------------------------------------
# Test 3c: route_after_supervisor fans out to 3 agents normally
# ---------------------------------------------------------------------------

def test_route_after_supervisor_fans_out():
    """Normal case: fan out to 3 data agents."""
    state = {"status": "in_progress"}

    result = route_after_supervisor(state)

    assert set(result) == {"sentiment_scout", "sec_auditor", "market_quant"}


# ---------------------------------------------------------------------------
# Test 3d: LLM call itself crashes → status="completed" (not a traceback)
# ---------------------------------------------------------------------------

@patch("app.agents.supervisor._resolve_ticker")
def test_supervisor_handles_llm_exception(mock_resolve):
    """If the LLM API itself throws, supervisor should catch and exit gracefully."""
    mock_resolve.side_effect = Exception("Gemini API rate limit exceeded")

    fake_state = {
        "user_query": "AAPL",
        "iteration_count": 0,
    }

    result = supervisor_node(fake_state)

    assert result.get("status") == "completed", "Should exit gracefully on LLM exception"


# ---------------------------------------------------------------------------
# Test 3e: Successful resolution → returns ticker in result
# ---------------------------------------------------------------------------

@patch("app.agents.supervisor._resolve_ticker")
def test_supervisor_resolves_ticker_successfully(mock_resolve):
    """On successful resolution, supervisor should return the ticker."""
    mock_resolve.return_value = TickerResolution(
        ticker="AAPL",
        company_name="Apple Inc.",
        reasoning="Direct ticker input",
        resolved=True,
    )

    fake_state = {
        "user_query": "apple",
        "iteration_count": 0,
    }

    result = supervisor_node(fake_state)

    assert result.get("ticker") == "AAPL"
    assert "messages" in result
    assert "status" not in result, "Should NOT set status=completed on success"
