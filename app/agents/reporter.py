from langchain_core.messages import AIMessage
from app.state import AlphaLensState, FinalReport
from app.config import get_llm


def report_generator_node(state: AlphaLensState) -> dict:
    """Aggregate all information and generate the final investment risk report"""
    print("📋 Report Generator: Generating final report...")

    ticker = state["ticker"]
    truth = state.get("truth_check_report")

    # Collect all Agent messages
    all_messages = "\n".join([m.content for m in state.get("messages", []) if hasattr(m, "content")])

    llm = get_llm()
    structured_llm = llm.with_structured_output(FinalReport)

    prompt = f"""You are a senior financial analyst writing a balanced investment risk report for {ticker}. Respond in English only.

=== Agent Analysis Records ===
{all_messages}

=== Cross-Validation Conclusion ===
{f"Consistency Score: {truth.overall_consistency}, Conflicts: {len(truth.conflicts)}, Summary: {truth.summary}" if truth else "No validation data"}

RISK LEVEL SCALE (1-10 integer, follow strictly):
- 1-2 (Very Low): Strong fundamentals, positive sentiment, stable technicals, no red flags. Think: top-tier blue-chips in a bull market (e.g., AAPL with strong earnings beat).
- 3-4 (Low): Mostly positive signals with minor concerns. Solid company with some short-term headwinds or slight overvaluation debate.
- 5 (Neutral): Genuinely mixed signals — roughly equal bullish and bearish indicators. This is the BASELINE for uncertainty.
- 6 (Slightly Elevated): More bearish signals than bullish, but no fundamental deterioration. Typical for a stock in a mild correction or facing temporary headwinds.
- 7 (Elevated): Clear bearish trend across sentiment AND technicals, with some fundamental concerns. Still a viable company but facing real challenges.
- 8 (High): Confirmed negative signals across ALL three dimensions. SEC red flags present, sustained price decline, and negative sentiment consensus.
- 9 (Very High): Severe fundamental deterioration (revenue decline, rising debt) combined with regulatory concerns or major governance issues.
- 10 (Critical): Active fraud investigation, SEC enforcement, imminent bankruptcy, or confirmed financial manipulation.

CALIBRATION GUIDANCE:
- Most S&P 500 stocks in normal conditions should land between 3-6.
- A stock with mixed sentiment but solid fundamentals should be 4-5, NOT 7-8.
- Short-term price dips or bearish social media chatter alone should NOT push above 6.
- If conflicts were unresolved after 2 rounds, this is normal market uncertainty — cap at 5-6 unless hard evidence says otherwise.

Fill in the following fields:
- ticker: "{ticker}"
- risk_level: integer 1-10 using the scale above
- executive_summary: 2-3 sentences covering BOTH key risks AND positive factors
- sentiment_section: balanced sentiment summary — include both bearish AND bullish voices (2-3 sentences)
- fundamentals_section: fundamentals summary — include strengths, not just risks (2-3 sentences)
- technical_section: technical summary — note both support levels and resistance (2-3 sentences)
- conflicts_section: conflicts summary — if unresolved, explain that uncertainty is normal

Be professional, balanced, and concise."""

    try:
        report = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"   ❌ Report generation failed: {e}")
        report = FinalReport(
            ticker=ticker,
            risk_level=5,
            executive_summary=f"Report generation failed: {str(e)}",
            sentiment_section="N/A",
            fundamentals_section="N/A",
            technical_section="N/A",
            conflicts_section="N/A",
        )

    print(f"   ✅ Final risk level: {report.risk_level}/10")

    return {"final_report": report, "status": "completed"}
