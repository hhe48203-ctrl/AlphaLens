from datetime import datetime



def save_detailed_report(state: dict, filename: str = "latest_report.md"):
    """Export the full analysis results as a Markdown file"""
    lines = []
    ticker = state.get("ticker", "UNKNOWN")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- Title --
    lines.append(f"# AlphaLens Report: {ticker}")
    lines.append(f"> Generated at {now} | Iterations: {state.get('iteration_count', 0)}")
    lines.append("")

    # -- Final report summary --
    final = state.get("final_report")
    if final:
        level = final.risk_level
        risk_bar = "█" * level + "░" * (10 - level)
        risk_labels = {1: "Very Low", 2: "Very Low", 3: "Low", 4: "Low", 5: "Neutral",
                       6: "Slightly Elevated", 7: "Elevated", 8: "High", 9: "Very High", 10: "Critical"}
        label = risk_labels.get(level, "Unknown")
        lines.append(f"## Risk Level: {level}/10 — {label}")
        lines.append(f"```")
        lines.append(f"[{risk_bar}] {level}/10")
        lines.append(f"```")
        lines.append("")
        lines.append(f"**Executive Summary:** {final.executive_summary}")
        lines.append("")
    else:
        lines.append("## ⚪ No Final Report Generated")
        lines.append("")

    # -- Market Quant data --
    quant = state.get("market_quant_report")
    lines.append("---")
    lines.append("## 📊 Market Quant Data")
    lines.append("")
    if quant:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Current Price | ${quant.current_price} |")
        lines.append(f"| 30-Day Change | {quant.price_change_30d_pct}% |")
        lines.append(f"| 30-Day Volatility | {quant.volatility_30d} |")
        lines.append(f"| Sharpe Ratio | {quant.sharpe_ratio} |")
        lines.append(f"| RSI-14 | {quant.rsi_14} |")
        lines.append(f"| Technical Signal | **{quant.technical_signal}** |")
        lines.append(f"| Confidence | {quant.confidence} |")
        lines.append("")
        lines.append(f"**LLM Analysis:** {quant.raw_data_summary}")
    else:
        lines.append("*No market data available.*")
    lines.append("")

    # -- Sentiment analysis --
    sentiment = state.get("sentiment_report")
    lines.append("---")
    lines.append("## 🔍 Sentiment Analysis")
    lines.append("")
    if sentiment:
        score = sentiment.sentiment_score
        if score > 0.3:
            mood = "📈 Bullish"
        elif score < -0.3:
            mood = "📉 Bearish"
        else:
            mood = "➡️ Neutral"

        lines.append(f"**Overall Mood:** {mood} (score: {score})")
        lines.append(f"**Discussion Volume Change:** {sentiment.volume_change_pct}%")
        lines.append(f"**Confidence:** {sentiment.confidence}")
        lines.append("")
        lines.append("**Key Narratives:**")
        for n in sentiment.key_narratives:
            lines.append(f"- {n}")
        lines.append("")
        lines.append("**Raw Evidence:**")
        for e in sentiment.raw_evidence:
            lines.append(f"- {e}")
    else:
        lines.append("*No sentiment data available.*")
    lines.append("")

    # -- SEC Filing --
    sec = state.get("sec_report")
    lines.append("---")
    lines.append("## ⚖️ SEC Filing Analysis")
    lines.append("")
    if sec:
        lines.append(f"**Filing Type:** {sec.filing_type}")
        lines.append(f"**Confidence:** {sec.confidence}")
        lines.append("")
        lines.append("**Risk Factors:**")
        for r in sec.risk_factors:
            lines.append(f"- {r}")
        lines.append("")
        if sec.red_flags:
            lines.append("**🚩 Red Flags:**")
            for r in sec.red_flags:
                lines.append(f"- ⚠️ {r}")
            lines.append("")
        if sec.key_financial_metrics:
            lines.append("**Financial Metrics:**")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for k, v in sec.key_financial_metrics.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
    else:
        lines.append("*No SEC data available.*")
    lines.append("")

    # -- Truth Check --
    truth = state.get("truth_check_report")
    lines.append("---")
    lines.append("## 🔎 Truth Check & Conflicts")
    lines.append("")
    if truth:
        lines.append(f"**Overall Consistency:** {truth.overall_consistency}")
        lines.append(f"**Recommendation:** {truth.recommendation}")
        lines.append(f"**Summary:** {truth.summary}")
        lines.append("")
        if truth.conflicts:
            lines.append(f"### Conflicts Detected: {len(truth.conflicts)}")
            lines.append("")
            for i, c in enumerate(truth.conflicts, 1):
                severity_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(c.severity, "⚪")
                lines.append(f"#### {severity_emoji} Conflict #{i} ({c.severity.upper()})")
                lines.append(f"- **{c.source_a}:** {c.claim_a}")
                lines.append(f"- **{c.source_b}:** {c.claim_b}")
                lines.append(f"- **Explanation:** {c.explanation}")
                lines.append("")
        else:
            lines.append("✅ No conflicts detected.")
    else:
        lines.append("*No truth check data available.*")
    lines.append("")

    # -- Detailed report sections --
    if final:
        lines.append("---")
        lines.append("## 📋 Detailed Report Sections")
        lines.append("")
        lines.append(f"### Sentiment\n{final.sentiment_section}")
        lines.append("")
        lines.append(f"### Fundamentals\n{final.fundamentals_section}")
        lines.append("")
        lines.append(f"### Technicals\n{final.technical_section}")
        lines.append("")
        lines.append(f"### Conflicts\n{final.conflicts_section}")
        lines.append("")

    # -- Agent message log --
    messages = state.get("messages", [])
    if messages:
        lines.append("---")
        lines.append("## 💬 Agent Message Log")
        lines.append("")
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                name = getattr(msg, "name", "unknown")
                lines.append(f"**[{name}]**")
                lines.append(f"> {msg.content}")
                lines.append("")

    # -- Write to file --
    content = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"📄 Detailed report saved to: {filename}")
