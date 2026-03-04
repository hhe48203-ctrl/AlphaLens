"""
Test 4: Sentiment Scout — API failure graceful handling

Tests that when Grok/Tavily APIs fail, the sentiment node either
handles the error gracefully or returns an errors list.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.agents.sentiment import sentiment_node, _search_x, _search_news
from app.state import SentimentReport


# ---------------------------------------------------------------------------
# Test 4a: Both X search and Tavily fail → LLM still gets called with fallback text
# ---------------------------------------------------------------------------

@patch("app.agents.sentiment.get_llm")
@patch("app.agents.sentiment._search_news")
@patch("app.agents.sentiment._search_x")
def test_sentiment_handles_both_api_failures(mock_x, mock_news, mock_llm):
    """When both X and Tavily fail, node should still call LLM with fallback strings."""
    mock_x.side_effect = Exception("Grok API key expired")
    mock_news.side_effect = Exception("Tavily rate limit")

    # Mock LLM to return a valid SentimentReport
    fake_report = SentimentReport(
        ticker="TSLA",
        sentiment_score=0.0,
        volume_change_pct=0.0,
        key_narratives=["Unable to assess due to data source failures"],
        raw_evidence=["X search failed", "News search failed"],
        confidence=0.2,
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = fake_report
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = mock_llm_instance

    fake_state = {"ticker": "TSLA"}

    result = sentiment_node(fake_state)

    # Should still produce a report (not crash)
    assert "sentiment_report" in result
    assert result["sentiment_report"].ticker == "TSLA"


# ---------------------------------------------------------------------------
# Test 4b: LLM structured output call fails → returns errors
# ---------------------------------------------------------------------------

@patch("app.agents.sentiment.get_llm")
@patch("app.agents.sentiment._search_news")
@patch("app.agents.sentiment._search_x")
def test_sentiment_returns_error_when_llm_fails(mock_x, mock_news, mock_llm):
    """When LLM call fails, node should return errors list."""
    mock_x.return_value = "Some X data"
    mock_news.return_value = "Some news data"

    # LLM throws
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.side_effect = Exception("LLM output parse error")
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = mock_llm_instance

    fake_state = {"ticker": "NVDA"}

    result = sentiment_node(fake_state)

    assert "errors" in result
    assert "Sentiment Scout LLM failed" in result["errors"][0]


# ---------------------------------------------------------------------------
# Test 4c: X search fails but Tavily succeeds → node still works
# ---------------------------------------------------------------------------

@patch("app.agents.sentiment.get_llm")
@patch("app.agents.sentiment._search_news")
@patch("app.agents.sentiment._search_x")
def test_sentiment_works_with_partial_data(mock_x, mock_news, mock_llm):
    """If only X fails but Tavily works, the node should still produce a report."""
    mock_x.side_effect = Exception("X API down")
    mock_news.return_value = "• AAPL beats earnings: Strong Q4 results..."

    fake_report = SentimentReport(
        ticker="AAPL",
        sentiment_score=0.6,
        volume_change_pct=15.0,
        key_narratives=["Strong earnings beat", "Positive analyst upgrades"],
        raw_evidence=["AAPL beats earnings article"],
        confidence=0.6,
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = fake_report
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = mock_llm_instance

    fake_state = {"ticker": "AAPL"}

    result = sentiment_node(fake_state)

    assert "sentiment_report" in result
    assert result["sentiment_report"].sentiment_score == 0.6
    # Confidence should reflect partial data
    assert result["sentiment_report"].confidence <= 0.7


# ---------------------------------------------------------------------------
# Test 4d: Sentiment score range validation via Pydantic
# ---------------------------------------------------------------------------

def test_sentiment_score_rejects_out_of_range():
    """Pydantic should reject sentiment_score outside [-1.0, 1.0]."""
    with pytest.raises(Exception):  # ValidationError
        SentimentReport(
            ticker="TEST",
            sentiment_score=1.5,  # Invalid! Must be <= 1.0
            volume_change_pct=0.0,
            key_narratives=[],
            raw_evidence=[],
            confidence=0.5,
        )

    with pytest.raises(Exception):
        SentimentReport(
            ticker="TEST",
            sentiment_score=-1.5,  # Invalid! Must be >= -1.0
            volume_change_pct=0.0,
            key_narratives=[],
            raw_evidence=[],
            confidence=0.5,
        )
