from langchain_core.messages import AIMessage
from app.state import AlphaLensState, TruthCheckReport
from app.config import get_llm


def truth_checker_node(state: AlphaLensState) -> dict:
    """
    Truth Checker Agent: cross-validates three agent reports to find contradictions.
    If major conflicts found, recommends looping back for re-investigation.
    """
    print("🔎 Truth Checker: Cross-validating agent reports...")

    iteration = state["iteration_count"]
    sentiment = state.get("sentiment_report")
    sec = state.get("sec_report")
    quant = state.get("market_quant_report")

    # Compile three reports into text for LLM analysis
    reports_text = ""

    if sentiment:
        reports_text += f"""
--- Sentiment Scout Report ---
Sentiment Score: {sentiment.sentiment_score} (-1 bearish ~ 1 bullish)
Discussion Volume Change: {sentiment.volume_change_pct}%
Key Narratives: {', '.join(sentiment.key_narratives)}
Confidence: {sentiment.confidence}
"""

    if quant:
        reports_text += f"""
--- Market Quant Report ---
Current Price: ${quant.current_price}
30-Day Change: {quant.price_change_30d_pct}%
Volatility: {quant.volatility_30d}
Sharpe Ratio: {quant.sharpe_ratio}
Technical Signal: {quant.technical_signal}
Confidence: {quant.confidence}
"""

    if sec:
        reports_text += f"""
--- SEC Auditor Report ---
Filing Type: {sec.filing_type}
Risk Factors: {', '.join(sec.risk_factors)}
Red Flags: {', '.join(sec.red_flags)}
Confidence: {sec.confidence}
"""

    if not reports_text.strip():
        print("   ⚠️ No reports received")
        report = TruthCheckReport(
            conflicts=[],
            overall_consistency=0.0,
            recommendation="needs_more_data",
            summary="No reports available for validation",
        )
        return {"truth_check_report": report, "iteration_count": iteration + 1}

    llm = get_llm()
    structured_llm = llm.with_structured_output(TruthCheckReport)

    prompt = f"""You are a balanced financial intelligence cross-validation expert. Below are reports from three independent Agents analyzing the same stock. Respond in English only.

{reports_text}

This is validation round {iteration + 1} (max 2 rounds).

IMPORTANT — Balanced Analysis Rules:
1. You MUST list at least 2 POSITIVE factors that support a bullish or stable outlook, even if the overall picture leans bearish. Every stock has strengths — find them.
2. Conflict Severity Guidelines (be strict):
   - "low": Differences in opinion or interpretation (e.g., one agent is mildly bearish while another is neutral). This is NORMAL and expected.
   - "medium": Meaningful disagreements on direction, but no factual contradictions (e.g., sentiment says bearish but technicals show recent uptrend). Treat short-term sentiment swings as MARKET NOISE, not real conflicts.
   - "high": Direct factual contradictions with large magnitude (e.g., one agent reports positive revenue growth while another cites revenue decline). Only use this for verifiable factual errors.
   - "critical": Evidence of fraud, manipulation, or SEC enforcement — extremely rare.
3. Do NOT treat normal market uncertainty as a conflict. Different agents having slightly different outlooks is healthy, not problematic.

Your task:
1. Compare the three reports and identify GENUINE contradictions (not mere differences in tone)
2. For each real contradiction, specify sources, claims, severity (using the strict guidelines above), and explanation
3. Provide an overall consistency score (0-1). If agents mostly agree on direction, score should be >= 0.7
4. In the summary field, explicitly mention both risks AND positive factors
5. recommendation field:
   - "consistent": ALL three agents agree on direction AND key facts with no notable disagreements
   - "minor_conflicts": some tone differences but no factual contradictions
   - "needs_more_data": use this in round 1 if ANY of the following are true: (a) agents disagree on the overall direction (one bullish, one bearish), (b) there is a medium or high severity conflict, (c) one agent's confidence is below 0.7, or (d) the data seems incomplete or contradictory enough to benefit from a second look. Prefer this over "consistent" in round 1 — a second round improves report quality.
   - "major_conflicts": genuine factual contradictions exist (round 2 only)

{"Round 2 Special Rule: Do NOT recommend needs_more_data. If conflicts remain unresolved, accept the uncertainty gracefully. Use 'minor_conflicts' or 'consistent' and note in summary that some information sources show irreconcilable differences, reflecting normal market uncertainty. Do NOT escalate risk just because of unresolved disagreements." if iteration >= 1 else "Round 1 Guidance: Lean toward 'needs_more_data' when in doubt — a second investigation round produces more reliable conclusions. Only use 'consistent' if all three agents clearly agree."}"""

    try:
        report = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        report = TruthCheckReport(
            conflicts=[],
            overall_consistency=0.5,
            recommendation="minor_conflicts",
            summary=f"Analysis error: {str(e)}",
        )

    conflict_count = len(report.conflicts)
    if conflict_count > 0:
        print(f"   ⚠️ Found {conflict_count} conflict(s) (recommendation: {report.recommendation})")
    else:
        print(f"   ✅ No major conflicts (recommendation: {report.recommendation})")

    return {
        "truth_check_report": report,
        "iteration_count": iteration + 1,
        "messages": [AIMessage(
            content=f"[Truth Checker] Consistency: {report.overall_consistency}, "
                    f"Conflicts: {conflict_count}, Recommendation: {report.recommendation}",
            name="truth_checker",
        )],
    }
