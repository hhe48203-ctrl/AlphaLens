"""
Agent Output Quality Eval (LLM-as-Judge)

Runs each agent with a real ticker (AAPL), then uses Gemini LLM to judge
the quality of the output along three dimensions: faithfulness, completeness,
and reasonableness. All tests call real external APIs.
"""

import pytest
from app.config import get_llm
from app.agents.sentiment import sentiment_node
from app.agents.market_quant import market_quant_node
from app.agents.sec_auditor import sec_auditor_node
from tests.eval.eval_schemas import AgentEvalResult


TICKER = "AAPL"


def _make_state(ticker: str = TICKER) -> dict:
    """Minimal state dict for running a single agent node."""
    return {
        "ticker": ticker,
        "iteration_count": 0,
        "sentiment_report": None,
        "sec_report": None,
        "market_quant_report": None,
        "errors": [],
    }


def _judge_agent(llm, agent_name: str, report_text: str, raw_evidence: str) -> AgentEvalResult:
    """Ask the LLM to judge an agent's output quality."""
    judge_llm = llm.with_structured_output(AgentEvalResult)

    prompt = f"""You are an expert evaluator of financial analysis agents. Score the following {agent_name} report on three dimensions (1-5 each).

REPORT:
{report_text}

RAW EVIDENCE / DATA:
{raw_evidence}

Scoring rubric:
- Faithfulness (1-5): Are the agent's qualitative conclusions (narratives, signals, risk assessments) supported by the evidence? Note: numerical metrics like price, RSI, volatility, and Sharpe ratio are COMPUTED from API data (yfinance, SEC EDGAR), so they count as grounded even if the raw_evidence field only contains a text summary. Only penalize for claims that are clearly fabricated or contradicted by the evidence. 5 = fully grounded, 1 = hallucinated.
- Completeness (1-5): Does the report cover the key dimensions expected for this type of analysis? 5 = thorough, 1 = major gaps.
- Reasonableness (1-5): Are numerical values in plausible ranges for real market data? 5 = all values sensible, 1 = clearly wrong.

Be fair but critical. A score of 3 means acceptable."""

    return judge_llm.invoke(prompt)


def _print_eval(agent_name: str, result: AgentEvalResult):
    """Print eval scores to stdout for pytest -v visibility."""
    print(f"\n{'='*60}")
    print(f"  {agent_name} Quality Eval")
    print(f"{'='*60}")
    print(f"  Faithfulness:   {result.faithfulness}/5 — {result.faithfulness_reasoning[:100]}")
    print(f"  Completeness:   {result.completeness}/5 — {result.completeness_reasoning[:100]}")
    print(f"  Reasonableness: {result.reasonableness}/5 — {result.reasonableness_reasoning[:100]}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Sentiment Scout
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sentiment_agent_quality():
    """Run Sentiment Scout on AAPL, then LLM-judge the output."""
    state = _make_state()
    result = sentiment_node(state)

    report = result.get("sentiment_report")
    assert report is not None, "Sentiment agent should produce a report"

    report_text = (
        f"Ticker: {report.ticker}\n"
        f"Sentiment Score: {report.sentiment_score}\n"
        f"Volume Change: {report.volume_change_pct}%\n"
        f"Key Narratives: {report.key_narratives}\n"
        f"Confidence: {report.confidence}"
    )
    raw_evidence = "\n".join(report.raw_evidence)

    llm = get_llm()
    eval_result = _judge_agent(llm, "Sentiment Scout", report_text, raw_evidence)
    _print_eval("Sentiment Scout", eval_result)

    assert eval_result.faithfulness >= 2, f"Faithfulness {eval_result.faithfulness} < 2"
    assert eval_result.completeness >= 2, f"Completeness {eval_result.completeness} < 2"
    assert eval_result.reasonableness >= 3, f"Reasonableness {eval_result.reasonableness} < 3"


# ---------------------------------------------------------------------------
# Market Quant
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_market_quant_agent_quality():
    """Run Market Quant on AAPL, then LLM-judge the output."""
    state = _make_state()
    result = market_quant_node(state)

    report = result.get("market_quant_report")
    assert report is not None, "Market Quant agent should produce a report"

    report_text = (
        f"Ticker: {report.ticker}\n"
        f"Current Price: ${report.current_price}\n"
        f"30-Day Change: {report.price_change_30d_pct}%\n"
        f"Volatility: {report.volatility_30d}\n"
        f"Sharpe Ratio: {report.sharpe_ratio}\n"
        f"RSI-14: {report.rsi_14}\n"
        f"Technical Signal: {report.technical_signal}\n"
        f"Confidence: {report.confidence}"
    )
    raw_evidence = report.raw_data_summary

    llm = get_llm()
    eval_result = _judge_agent(llm, "Market Quant", report_text, raw_evidence)
    _print_eval("Market Quant", eval_result)

    assert eval_result.faithfulness >= 2, f"Faithfulness {eval_result.faithfulness} < 2"
    assert eval_result.completeness >= 3, f"Completeness {eval_result.completeness} < 3"
    assert eval_result.reasonableness >= 3, f"Reasonableness {eval_result.reasonableness} < 3"


# ---------------------------------------------------------------------------
# SEC Auditor
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sec_auditor_agent_quality():
    """Run SEC Auditor on AAPL, then LLM-judge the output."""
    state = _make_state()
    result = sec_auditor_node(state)

    report = result.get("sec_report")
    assert report is not None, "SEC Auditor agent should produce a report"

    report_text = (
        f"Ticker: {report.ticker}\n"
        f"Filing Type: {report.filing_type}\n"
        f"Risk Factors: {report.risk_factors}\n"
        f"Key Financial Metrics: {report.key_financial_metrics}\n"
        f"Red Flags: {report.red_flags}\n"
        f"Confidence: {report.confidence}"
    )
    raw_evidence = "\n".join(report.raw_evidence)

    llm = get_llm()
    eval_result = _judge_agent(llm, "SEC Auditor", report_text, raw_evidence)
    _print_eval("SEC Auditor", eval_result)

    assert eval_result.faithfulness >= 2, f"Faithfulness {eval_result.faithfulness} < 2"
    assert eval_result.completeness >= 2, f"Completeness {eval_result.completeness} < 2"
    assert eval_result.reasonableness >= 2, f"Reasonableness {eval_result.reasonableness} < 2"
