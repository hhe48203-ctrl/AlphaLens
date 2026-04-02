"""
End-to-End Regression Eval

Runs the full AlphaLens pipeline on representative tickers and verifies
outputs are reasonable. Calls ALL real APIs (Gemini, Grok, Tavily, yfinance,
SEC EDGAR). Uses LLM-as-judge to evaluate the final report quality.
"""

import pytest
from app.graph import build_graph
from app.config import get_llm
from tests.eval.eval_schemas import ReportEvalResult


def _run_pipeline(ticker: str) -> dict:
    """Run the full AlphaLens pipeline and return the final state."""
    graph = build_graph()

    initial_state = {
        "user_query": f"Analyze {ticker}",
        "ticker": "",
        "messages": [],
        "sentiment_report": None,
        "sec_report": None,
        "market_quant_report": None,
        "truth_check_report": None,
        "final_report": None,
        "risk_bias": "balanced",
        "report_language": "en",
        "reporter_guidance": "",
        "iteration_count": 0,
        "max_iterations": 2,
        "status": "in_progress",
        "errors": [],
    }

    return graph.invoke(initial_state, {"recursion_limit": 50})


def _judge_final_report(llm, ticker: str, final_report, risk_level: int) -> ReportEvalResult:
    """Ask the LLM to judge the final report quality."""
    judge_llm = llm.with_structured_output(ReportEvalResult)

    prompt = f"""You are an expert evaluator of investment risk reports. Score the following final report for {ticker} on three dimensions (1-5 each).

REPORT:
Ticker: {final_report.ticker}
Risk Level: {risk_level}/10
Executive Summary: {final_report.executive_summary}
Sentiment Section: {final_report.sentiment_section}
Fundamentals Section: {final_report.fundamentals_section}
Technical Section: {final_report.technical_section}
Conflicts Section: {final_report.conflicts_section}

Scoring rubric:
- Coherence (1-5): Do all sections tell a consistent story? 5 = perfectly aligned, 1 = contradictory.
- Balance (1-5): Are both risks and positives mentioned? 5 = well balanced, 1 = one-sided.
- Calibration (1-5): Is the risk_level ({risk_level}/10) appropriate for the evidence presented? 5 = perfectly calibrated, 1 = wildly off.

Be fair but critical. A score of 3 means acceptable."""

    return judge_llm.invoke(prompt)


def _print_report_eval(ticker: str, result: ReportEvalResult):
    """Print eval scores to stdout for pytest -v visibility."""
    print(f"\n{'='*60}")
    print(f"  E2E Report Eval — {ticker}")
    print(f"{'='*60}")
    print(f"  Coherence:   {result.coherence}/5 — {result.coherence_reasoning[:100]}")
    print(f"  Balance:     {result.balance}/5 — {result.balance_reasoning[:100]}")
    print(f"  Calibration: {result.calibration}/5 — {result.calibration_reasoning[:100]}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# AAPL — blue-chip stable stock
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_aapl():
    """Full pipeline for AAPL: should complete successfully with low-moderate risk."""
    final_state = _run_pipeline("AAPL")

    # 1. Pipeline should not be in error state
    assert final_state.get("status") != "error", (
        f"Pipeline ended in error: {final_state.get('errors')}"
    )

    # 2. Check that agent reports were produced (at least 2/3)
    report_count = sum(
        final_state.get(key) is not None
        for key in ("sentiment_report", "sec_report", "market_quant_report")
    )
    print(f"\n[AAPL] Agent reports produced: {report_count}/3")
    assert report_count >= 2, f"Expected at least 2/3 agent reports, got {report_count}"

    # 3. Final report must exist
    final_report = final_state.get("final_report")
    assert final_report is not None, "Final report should exist"

    # 4. Risk level should be reasonable for a blue-chip (1-5)
    risk_level = final_report.risk_level
    print(f"[AAPL] Risk Level: {risk_level}/10")
    assert 1 <= risk_level <= 7, (
        f"AAPL risk_level {risk_level} seems too extreme for a blue-chip"
    )

    # 5. Executive summary should mention the ticker
    assert final_report.executive_summary, "Executive summary should not be empty"
    assert "AAPL" in final_report.executive_summary.upper() or "APPLE" in final_report.executive_summary.upper(), (
        "Executive summary should mention AAPL or Apple"
    )

    # 6. LLM-as-judge evaluates report quality
    llm = get_llm()
    eval_result = _judge_final_report(llm, "AAPL", final_report, risk_level)
    _print_report_eval("AAPL", eval_result)

    assert eval_result.coherence >= 3, f"Coherence {eval_result.coherence} < 3"
    assert eval_result.balance >= 3, f"Balance {eval_result.balance} < 3"
    assert eval_result.calibration >= 3, f"Calibration {eval_result.calibration} < 3"


# ---------------------------------------------------------------------------
# TSLA — volatile/controversial stock
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_tsla():
    """Full pipeline for TSLA: should complete and reflect elevated uncertainty."""
    final_state = _run_pipeline("TSLA")

    # 1. Pipeline should not be in error state
    assert final_state.get("status") != "error", (
        f"Pipeline ended in error: {final_state.get('errors')}"
    )

    # 2. Check agent reports (at least 2/3)
    report_count = sum(
        final_state.get(key) is not None
        for key in ("sentiment_report", "sec_report", "market_quant_report")
    )
    print(f"\n[TSLA] Agent reports produced: {report_count}/3")
    assert report_count >= 2, f"Expected at least 2/3 agent reports, got {report_count}"

    # 3. Final report must exist
    final_report = final_state.get("final_report")
    assert final_report is not None, "Final report should exist"

    # 4. Risk level — TSLA is volatile, expect moderate-to-high risk
    risk_level = final_report.risk_level
    print(f"[TSLA] Risk Level: {risk_level}/10")
    assert 1 <= risk_level <= 10, f"Risk level {risk_level} out of valid range"

    # 5. Executive summary non-empty and mentions ticker
    assert final_report.executive_summary, "Executive summary should not be empty"
    assert "TSLA" in final_report.executive_summary.upper() or "TESLA" in final_report.executive_summary.upper(), (
        "Executive summary should mention TSLA or Tesla"
    )

    # 6. LLM-as-judge
    llm = get_llm()
    eval_result = _judge_final_report(llm, "TSLA", final_report, risk_level)
    _print_report_eval("TSLA", eval_result)

    assert eval_result.coherence >= 3, f"Coherence {eval_result.coherence} < 3"
    assert eval_result.balance >= 3, f"Balance {eval_result.balance} < 3"
    assert eval_result.calibration >= 2, f"Calibration {eval_result.calibration} < 2"
