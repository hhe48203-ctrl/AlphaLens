"""
AlphaLens — Streamlit Web Interface
Run: streamlit run app_ui.py
"""
import streamlit as st
import os
import sys
import io
import uuid
from datetime import datetime
from dotenv import load_dotenv

# -- Environment init (must be before Streamlit components) --
load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "AlphaLens")

from app.graph import build_graph
from app.reporter import save_detailed_report

# -- Page config --
st.set_page_config(
    page_title="AlphaLens — AI Risk Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Global styles --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Base */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { max-width: 1100px; padding-top: 3rem; }

    /* Fix: hide "Press Enter to apply" tooltip to prevent overlap with placeholder */
    .stTextInput [data-testid="InputInstructions"] { display: none; }

    /* Risk gauge */
    .risk-gauge {
        text-align: center;
        padding: 1.5rem 1rem;
        border-radius: 16px;
        border: 1px solid;
    }
    .risk-gauge .score { font-size: 3.2rem; font-weight: 700; line-height: 1; }
    .risk-gauge .label { font-size: 1.1rem; font-weight: 600; margin-top: 0.3rem; }
    .risk-gauge .bar { font-family: monospace; font-size: 1rem; margin-top: 0.6rem; letter-spacing: 2px; opacity: 0.8; }

    /* Metric card */
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid;
    }
    .metric-card .metric-label { font-size: 0.8rem; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .metric-value { font-size: 1.4rem; font-weight: 700; color: #1f2937; }
    .metric-card .metric-delta { font-size: 0.85rem; font-weight: 500; }

    /* Terminal log */
    .log-line { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 0.82rem; line-height: 1.7; padding: 2px 0; }
    .log-container { background: #0d1117; color: #c9d1d9; padding: 1rem 1.2rem; border-radius: 10px; max-height: 400px; overflow-y: auto; }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════

class LogBuffer:
    """Thread-safe log buffer — only collects print output (no Streamlit calls)"""

    def __init__(self):
        self.lines = []

    def write(self, text):
        if text:  # Preserve newlines, don't strip()
            self.lines.append(text)

    def flush(self):
        pass

    def get_logs(self) -> str:
        return "".join(self.lines)


def update_ui_from_logs(logs: str, log_area, status_text):
    """Update Streamlit UI from main thread based on log content (safe to call)"""
    # Update status text (match the latest state)
    if "Data Quality Gate" in logs and "All data agents failed" in logs:
        status_text.write("**❌ Analysis failed: all data agents failed**")
    elif "Report Generator" in logs:
        status_text.write("**📋 Generating final report...**")
    elif "Round 2" in logs or "round 2" in logs:
        status_text.write("**⚠️ Conflicts detected, launching round 2 investigation...**")
    elif "Truth Checker" in logs:
        status_text.write("**🔎 Cross-validating agent reports...**")
    elif "Could not identify" in logs:
        status_text.write("**❌ Could not identify target stock**")
    elif "Resolved" in logs:
        status_text.write("**📡 Agents collecting data in parallel...**")

    # Render colored log
    html = colorize_log(logs)
    log_area.markdown(
        f'<div class="log-container">{html}</div>',
        unsafe_allow_html=True,
    )


def colorize_log(text: str) -> str:
    """Convert agent terminal logs to color-coded HTML"""
    color_map = {
        "📊": "#58a6ff",  # Market Quant - blue
        "🔍": "#d2a8ff",  # Sentiment - purple
        "⚖️": "#7ee787",  # SEC Auditor - green
        "🔎": "#ffa657",  # Truth Checker - orange
        "🧠": "#ff7b72",  # Supervisor - red
        "📋": "#79c0ff",  # Reporter - light blue
        "✅": "#7ee787",
        "❌": "#ff7b72",
        "⚠️": "#ffa657",
        "⏩": "#8b949e",
    }
    lines_html = []
    for line in text.strip().split("\n"):
        color = "#c9d1d9"  # default gray-white
        for emoji, c in color_map.items():
            if emoji in line:
                color = c
                break
        escaped = line.replace("<", "&lt;").replace(">", "&gt;")
        lines_html.append(f'<div class="log-line" style="color:{color};">{escaped}</div>')
    return "\n".join(lines_html)


def get_risk_color(level: int) -> str:
    """Return color hex based on risk level"""
    if level <= 3:
        return "#22c55e"
    elif level <= 5:
        return "#eab308"
    elif level <= 7:
        return "#f97316"
    else:
        return "#ef4444"


RISK_LABELS = {
    1: "Very Low", 2: "Very Low", 3: "Low", 4: "Low", 5: "Neutral",
    6: "Slightly Elevated", 7: "Elevated", 8: "High", 9: "Very High", 10: "Critical"
}

REVIEW_LABELS = {
    "conservative": "Conservative",
    "balanced": "Balanced",
    "aggressive": "Aggressive",
}


# ════════════════════════════════════════════
# Render functions
# ════════════════════════════════════════════

def render_risk_gauge(level: int):
    """Render risk level gauge widget"""
    color = get_risk_color(level)
    label = RISK_LABELS.get(level, "Unknown")
    bar = "█" * level + "░" * (10 - level)
    st.markdown(f"""
    <div class="risk-gauge" style="background: linear-gradient(135deg, {color}12, {color}06); border-color: {color}40;">
        <div class="score" style="color: {color};">{level}/10</div>
        <div class="label" style="color: {color};">{label}</div>
        <div class="bar">[{bar}]</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric(label: str, value: str, delta: str = "", color: str = "#3b82f6"):
    """Render a custom metric card"""
    delta_html = ""
    if delta:
        delta_color = "#22c55e" if not delta.startswith("-") else "#ef4444"
        delta_html = f'<div class="metric-delta" style="color: {delta_color};">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_agent_messages(messages):
    """Render agent reasoning chain as chat messages"""
    agent_icons = {
        "supervisor": "🧠", "market_quant": "📊", "sentiment_scout": "🔍",
        "sec_auditor": "⚖️", "truth_checker": "🔎", "report_generator": "📋",
    }
    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            name = getattr(msg, "name", "unknown")
            icon = agent_icons.get(name, "💬")
            with st.chat_message(name, avatar=icon):
                st.markdown(msg.content[:600])


def build_initial_state(query: str) -> dict:
    return {
        "user_query": query,
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


def clear_analysis_session():
    for key in [
        "result",
        "logs",
        "review_pending",
        "pending_state",
        "thread_id",
        "graph",
        "graph_config",
    ]:
        st.session_state.pop(key, None)


def run_graph_phase(graph, config: dict, status_text, log_area, initial_state: dict | None = None):
    """Run the graph until completion or an interrupt point."""
    log_buffer = LogBuffer()
    old_stdout = sys.stdout
    sys.stdout = log_buffer
    latest_result = initial_state.copy() if initial_state else {}

    try:
        for event in graph.stream(initial_state, config=config):
            for _, output in event.items():
                if isinstance(output, dict):
                    latest_result.update(output)
            update_ui_from_logs(log_buffer.get_logs(), log_area, status_text)
    finally:
        sys.stdout = old_stdout

    snapshot = graph.get_state(config)
    if snapshot and getattr(snapshot, "values", None):
        latest_result = dict(snapshot.values)

    return latest_result, log_buffer.get_logs(), snapshot


def render_review_panel():
    """Render the human review controls before report generation."""
    pending_state = st.session_state["pending_state"]
    truth = pending_state.get("truth_check_report")
    current_bias = pending_state.get("risk_bias", "balanced")
    current_language = pending_state.get("report_language", "en")
    current_guidance = pending_state.get("reporter_guidance", "")

    st.markdown("## Human Review")
    st.caption("Review the truth-check output and adjust final report settings before generating the report.")

    if truth:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Overall Consistency", f"{truth.overall_consistency}")
        with c2:
            st.metric("Recommendation", truth.recommendation)
        st.markdown(f"**Summary:** {truth.summary}")

        if truth.conflicts:
            st.markdown(f"### Conflicts: {len(truth.conflicts)}")
            for i, conflict in enumerate(truth.conflicts, 1):
                severity_icon = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🟠",
                    "critical": "🔴",
                }.get(conflict.severity, "⚪")
                with st.expander(f"{severity_icon} Conflict #{i} — {conflict.severity.upper()}", expanded=(i == 1)):
                    st.markdown(f"**{conflict.source_a}:** {conflict.claim_a}")
                    st.markdown(f"**{conflict.source_b}:** {conflict.claim_b}")
                    st.info(conflict.explanation)
        else:
            st.success("✅ No conflicts detected")

    with st.form("human_review_form"):
        risk_bias = st.radio(
            "Risk scoring bias",
            options=list(REVIEW_LABELS.keys()),
            format_func=lambda value: REVIEW_LABELS[value],
            index=list(REVIEW_LABELS.keys()).index(current_bias),
            horizontal=True,
            help="Conservative tends to score mixed evidence slightly higher risk, aggressive slightly lower, balanced stays neutral.",
        )
        report_language = st.radio(
            "Report language",
            options=["en", "zh"],
            format_func=lambda value: "English" if value == "en" else "中文",
            index=0 if current_language == "en" else 1,
            horizontal=True,
        )
        reporter_guidance = st.text_area(
            "Custom guidance for reporter (optional)",
            value=current_guidance,
            placeholder="Example: focus more on downside risks from SEC disclosures; keep the summary concise.",
            height=120,
        )

        generate_report = st.form_submit_button("🚀 Generate Final Report", use_container_width=True)

    if generate_report:
        graph = st.session_state["graph"]
        config = st.session_state["graph_config"]
        graph.update_state(
            config,
            {
                "risk_bias": risk_bias,
                "report_language": report_language,
                "reporter_guidance": reporter_guidance.strip(),
            },
            as_node="truth_checker",
        )

        with st.status("📋 Finalizing report with human-reviewed settings...", expanded=True) as status:
            status_text = st.empty()
            status_text.write("**📋 Generating final report...**")
            log_area = st.empty()

            result, resume_logs, _ = run_graph_phase(graph, config, status_text, log_area)
            full_logs = st.session_state.get("logs", "") + resume_logs

            if not result.get("final_report"):
                status.update(label="❌ Final report generation failed", state="error")
                status_text.write("**❌ Final report generation failed**")
                update_ui_from_logs(full_logs, log_area, status_text)
                return

            status.update(label=f"✅ {result['ticker']} analysis complete", state="complete")
            status_text.write(f"**✅ {result['ticker']} analysis complete**")

        st.session_state["result"] = result
        st.session_state["logs"] = full_logs
        st.session_state["review_pending"] = False
        st.session_state.pop("pending_state", None)
        st.rerun()


# ════════════════════════════════════════════
# Main interface
# ════════════════════════════════════════════

def main():
    # -- Sidebar --
    with st.sidebar:
        st.markdown("## 🔍 AlphaLens")
        st.caption("Multi-Agent Investment Risk Intelligence")
        st.divider()

        query = st.text_input(
            "Query",
            placeholder="NVDA, Apple, pick a defense stock...",
            help="Supports ticker symbols, company names, or natural language descriptions",
            key="query_input",
        )

        run_clicked = st.button(
            "🚀 Start Analysis",
            type="primary",
            use_container_width=True,
            disabled=(not query.strip()),
        )

        st.divider()

        with st.expander("⚙️ API Keys", expanded=False):
            st.caption("Loaded from .env by default, override here if needed")
            g_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", "")[:8] + "..." if os.getenv("GOOGLE_API_KEY") else "", type="password", key="g_key")
            x_key = st.text_input("XAI API Key", value=os.getenv("XAI_API_KEY", "")[:8] + "..." if os.getenv("XAI_API_KEY") else "", type="password", key="x_key")
            t_key = st.text_input("Tavily API Key", value=os.getenv("TAVILY_API_KEY", "")[:8] + "..." if os.getenv("TAVILY_API_KEY") else "", type="password", key="t_key")

        st.divider()
        st.caption("Powered by LangGraph · Gemini · Grok · Tavily")

    # -- Main area --

    if st.session_state.get("review_pending") and not run_clicked:
        render_review_panel()
        return

    # Show cached results if available
    if "result" in st.session_state and not run_clicked:
        _render_results(st.session_state["result"], st.session_state["logs"])
        return

    if not run_clicked:
        # Landing page
        st.markdown("# 🔍 AlphaLens")
        st.markdown("**AI-Powered Multi-Agent Investment Risk Analysis**")
        st.markdown("")

        cols = st.columns(3)
        with cols[0]:
            st.markdown("##### 📊 Market Quant")
            st.caption("yfinance quantitative data, RSI-14, Sharpe ratio, volatility analysis")
        with cols[1]:
            st.markdown("##### 🔍 Sentiment Scout")
            st.caption("Grok X Search real-time social sentiment + Tavily news aggregation")
        with cols[2]:
            st.markdown("##### ⚖️ SEC Auditor")
            st.caption("SEC EDGAR 10-K/20-F filings + risk factor extraction")

        st.divider()
        st.info("👈 Enter a stock ticker, company name, or description in the sidebar and click **Start Analysis**")
        return

    clear_analysis_session()

    # -- Run analysis (streaming output) --
    with st.status("🧠 AlphaLens multi-agent analysis in progress...", expanded=True) as status:
        status_text = st.empty()
        status_text.write("**🔍 Resolving target stock...**")
        log_area = st.empty()

        graph = build_graph(enable_human_review=True)
        initial_state = build_initial_state(query)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        result, logs, snapshot = run_graph_phase(graph, config, status_text, log_area, initial_state)

        if not result.get("final_report"):
            if snapshot and getattr(snapshot, "next", None):
                status.update(label="⏸️ Waiting for human review", state="running")
                status_text.write("**⏸️ Review settings below before generating the final report**")
                update_ui_from_logs(logs, log_area, status_text)

                st.session_state["review_pending"] = True
                st.session_state["pending_state"] = result
                st.session_state["logs"] = logs
                st.session_state["thread_id"] = thread_id
                st.session_state["graph"] = graph
                st.session_state["graph_config"] = config
                render_review_panel()
                return

            if result.get("status") == "error":
                status.update(label="❌ Analysis failed: insufficient agent data", state="error")
                errors = result.get("errors", [])
                if errors:
                    status_text.write(f"**❌ Analysis failed: {errors[-1]}**")
                else:
                    status_text.write("**❌ Analysis failed due to upstream data errors**")
            else:
                status.update(label="❌ Could not identify target stock", state="error")
                status_text.write("**❌ Could not identify target stock, please try again**")
            update_ui_from_logs(logs, log_area, status_text)
            return

        ticker = result["ticker"]
        iterations = result.get("iteration_count", 1)

        if iterations > 1:
            st.warning(f"⚠️ Agent report conflicts detected, launched round 2 deep investigation ({iterations} rounds total)")

        status_text.write(f"**✅ {ticker} analysis complete**")
        status.update(label=f"✅ {ticker} analysis complete", state="complete")

    # Cache results
    st.session_state["result"] = result
    st.session_state["logs"] = logs

    _render_results(result, logs)


def _render_results(result: dict, logs: str):
    """Render analysis results"""
    report = result["final_report"]
    ticker = result["ticker"]
    iterations = result.get("iteration_count", 1)
    quant = result.get("market_quant_report")
    sentiment = result.get("sentiment_report")
    sec = result.get("sec_report")
    truth = result.get("truth_check_report")

    # -- Top: risk gauge + summary --
    st.divider()
    col_gauge, col_summary = st.columns([1, 2.5])

    with col_gauge:
        render_risk_gauge(report.risk_level)

    with col_summary:
        st.markdown(f"### {ticker}")
        st.markdown(report.executive_summary)
        st.caption(f"Iterations: {iterations} | Status: {result.get('status', 'N/A')}")

    st.divider()

    # -- Three-column data --
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 Market Quant")
        if quant:
            render_metric("Price", f"${quant.current_price}", f"{quant.price_change_30d_pct}%", "#3b82f6")
            render_metric("RSI-14", f"{quant.rsi_14}", "Oversold" if quant.rsi_14 < 30 else ("Overbought" if quant.rsi_14 > 70 else "Normal"), "#3b82f6")
            render_metric("Sharpe Ratio", f"{quant.sharpe_ratio}", "", "#3b82f6")
            render_metric("Volatility", f"{quant.volatility_30d}", "", "#3b82f6")
            signal_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(quant.technical_signal, "⚪")
            st.markdown(f"**Signal:** {signal_icon} {quant.technical_signal.upper()}")
        else:
            st.caption("No data available")

    with col2:
        st.markdown("#### 🔍 Sentiment")
        if sentiment:
            score = sentiment.sentiment_score
            mood = "📈 Bullish" if score > 0.3 else ("📉 Bearish" if score < -0.3 else "➡️ Neutral")
            s_color = "#22c55e" if score > 0.3 else ("#ef4444" if score < -0.3 else "#eab308")
            render_metric("Sentiment Score", f"{score}", mood, s_color)
            render_metric("Volume Change", f"{sentiment.volume_change_pct}%", "", "#8b5cf6")
            render_metric("Confidence", f"{sentiment.confidence}", "", "#8b5cf6")
            if sentiment.key_narratives:
                st.markdown("**Key Narratives:**")
                for n in sentiment.key_narratives[:3]:
                    st.caption(f"• {n[:100]}")
        else:
            st.caption("No data available")

    with col3:
        st.markdown("#### ⚖️ SEC Filing")
        if sec:
            render_metric("Filing Type", sec.filing_type, "", "#10b981")
            render_metric("Risk Factors", str(len(sec.risk_factors)), "", "#10b981")
            rf_count = len(sec.red_flags)
            rf_color = "#ef4444" if rf_count > 0 else "#10b981"
            render_metric("Red Flags", str(rf_count), "", rf_color)
            render_metric("Confidence", f"{sec.confidence}", "", "#10b981")
            if sec.red_flags:
                for f in sec.red_flags[:2]:
                    st.warning(f[:120], icon="🚩")
        else:
            st.caption("No data available")

    st.divider()

    # -- Detail tabs --
    tab1, tab2, tab3 = st.tabs(["📋 Detailed Report", "💬 Agent Chain", "🔎 Conflict Detection"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Sentiment Analysis")
            st.markdown(report.sentiment_section)
            st.markdown("##### Fundamentals")
            st.markdown(report.fundamentals_section)
        with col_b:
            st.markdown("##### Technical Analysis")
            st.markdown(report.technical_section)
            st.markdown("##### Conflicts & Uncertainty")
            st.markdown(report.conflicts_section)

    with tab2:
        messages = result.get("messages", [])
        if messages:
            render_agent_messages(messages)
        else:
            st.caption("No messages recorded")

    with tab3:
        if truth:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Overall Consistency", f"{truth.overall_consistency}")
            with c2:
                st.metric("Recommendation", truth.recommendation)
            st.markdown(f"**Summary:** {truth.summary}")

            if truth.conflicts:
                st.markdown(f"### Conflicts: {len(truth.conflicts)}")
                for i, c in enumerate(truth.conflicts, 1):
                    severity_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(c.severity, "⚪")
                    with st.expander(f"{severity_color} Conflict #{i} — {c.severity.upper()}", expanded=True):
                        st.markdown(f"**{c.source_a}:** {c.claim_a}")
                        st.markdown(f"**{c.source_b}:** {c.claim_b}")
                        st.info(f"💡 {c.explanation}")
            else:
                st.success("✅ No conflicts detected")
        else:
            st.caption("No validation data available")

    st.divider()

    # -- Download report --
    save_detailed_report(result)
    try:
        with open("latest_report.md", "r", encoding="utf-8") as f:
            report_md = f.read()
    except FileNotFoundError:
        report_md = "Report file not found."

    st.download_button(
        label="📥 Download Full Report (Markdown)",
        data=report_md,
        file_name=f"AlphaLens_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
