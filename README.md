<div align="center">

# 🔍 AlphaLens

**Multi-Agent Financial Intelligence System**

*Enter a stock ticker. Get a structured risk report.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[中文文档](README_CN.md)

</div>

---

## What is AlphaLens?

AlphaLens orchestrates **6 specialized AI agents** via [LangGraph](https://github.com/langchain-ai/langgraph) to analyze investment risk from three dimensions — **social sentiment**, **SEC filings**, and **quantitative indicators** — then cross-validates findings and generates a structured risk report.

> Built a LangGraph multi-agent system orchestrating 6 specialized AI agents with cyclic graph routing, Pydantic structured outputs, and SEC filing analysis to synthesize financial signals into risk reports, tracked via LangSmith.

<img width="1159" height="711" alt="Screenshot 2026-02-28 at 19 31 52" src="https://github.com/user-attachments/assets/a14d1c45-2bd2-45c2-8f8e-3953cccdd771" />

### Key Highlights

- 🔄 **Cyclic Graph** — Truth Checker can loop agents back for deeper investigation when conflicts are detected
- ⚡ **Parallel Fan-out** — 3 data agents run concurrently, fan-in at Truth Checker
- 🧩 **Structured Output** — Every agent returns Pydantic-validated JSON schemas
- 🧠 **Smart Ticker Resolution** — Natural language input ("apple", "biggest EV company") → resolved stock ticker
- 🖥️ **Streamlit UI** — Real-time streaming logs, risk gauge, metric cards, downloadable reports
- 📊 **LangSmith Tracing** — Full execution visibility in the LangSmith dashboard
- 💰 **$0 Total Cost** — All APIs used have generous free tiers

<img width="1044" height="808" alt="Screenshot 2026-02-28 at 19 34 36" src="https://github.com/user-attachments/assets/6156735e-62ff-4ffc-b6e3-b5b50d308b99" />

---

## Architecture

```
User Query (natural language)
         │
         ▼
   ┌─────────────┐
   │  Supervisor │ ← LLM-based ticker resolution
   └──────┬──────┘
          │
    ┌─────┼─────┐          Fan-out (parallel)
    ▼     ▼     ▼
┌──────┐┌──────┐┌──────┐
│Senti-││Market││ SEC  │
│ment  ││Quant ││Audit-│
│Scout ││      ││ or   │
└──┬───┘└──┬───┘└──┬───┘
   │       │       │
   └───────┼───────┘       Fan-in
           ▼
   ┌──────────────┐
   │ Truth Checker| ← Cross-validation
   └──────┬───────┘
          │
    ┌─────┴─────┐
    ▼           ▼
Conflicts?   No conflicts
    │           │
    ▼           ▼
Loop back   ┌──────────┐
to Super-   │  Report  │
visor       │Generator │
            └──────────┘
```

---

## Agents

| Agent | Role | Data Source |
|:------|:-----|:-----------|
| 🧠 **Supervisor** | Resolves user intent → ticker, dispatches tasks, manages re-investigation rounds | Gemini LLM |
| 🔍 **Sentiment Scout** | Analyzes real-time social media sentiment and news coverage | Grok `x_search` + Tavily |
| 📊 **Market Quant** | Computes price action, volatility, Sharpe ratio, RSI-14, volume trends | yfinance |
| ⚖️ **SEC Auditor** | Fetches SEC EDGAR filings (10-K / 20-F), extracts risk factors and red flags | SEC EDGAR + Tavily |
| 🔎 **Truth Checker** | Cross-validates 3 agent reports, identifies contradictions, recommends action | Gemini LLM |
| 📋 **Report Generator** | Synthesizes all findings into a calibrated 1-10 risk assessment | Gemini LLM |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/AlphaLens.git
cd AlphaLens
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
XAI_API_KEY=your_grok_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key   # optional, for tracing
```

### 3. Run

**Web UI (recommended):**

```bash
streamlit run app_ui.py
```

**CLI mode:**

```bash
python main.py
```

---

## Project Structure

```
AlphaLens/
├── app_ui.py                 # Streamlit web interface
├── main.py                   # CLI entry point
├── requirements.txt
├── .env                      # API keys (not committed)
│
├── app/
│   ├── config.py             # LLM configuration (Gemini 2.5 Flash)
│   ├── state.py              # Pydantic schemas + LangGraph state
│   ├── graph.py              # Graph builder with routing logic
│   ├── reporter.py           # Markdown report generator
│   │
│   └── agents/
│       ├── supervisor.py     # Task dispatch + ticker resolution
│       ├── sentiment.py      # X search + Tavily news sentiment
│       ├── market_quant.py   # yfinance quantitative analysis
│       ├── sec_auditor.py    # SEC EDGAR filing analysis
│       ├── truth_checker.py  # Cross-validation + conflict detection
│       └── reporter.py       # Final report generation (LLM)
│
├── tests/
│   ├── test_*.py             # Unit tests (routing, error handling, Pydantic validation)
│   └── eval/                 # 3-layer evaluation framework
│       ├── eval_schemas.py   # LLM judge Pydantic schemas
│       ├── test_agent_quality.py      # Layer 1: LLM-as-judge per agent
│       ├── test_truth_checker_eval.py # Layer 2: conflict detection accuracy
│       └── test_e2e_regression.py     # Layer 3: full pipeline regression
```

---

## UI Preview

The Streamlit interface provides:

- **Sidebar** — Query input, API key management, start button
- **Status Panel** — Real-time streaming logs with color-coded agent output
- **Risk Gauge** — 1-10 risk score with color-coded progress bar
- **Metric Cards** — Three-column layout showing Market / Sentiment / SEC data
- **Detail Tabs** — Full report, agent reasoning chain, conflict analysis
- **Download** — Export complete Markdown report

---

## Testing & Evaluation

AlphaLens includes a 3-layer evaluation framework that measures agent output quality, not just functional correctness.

```bash
# Run all eval tests (requires API keys + sufficient Gemini quota)
pytest tests/eval/ -v -s -m slow

# Run unit tests only (no API calls)
pytest tests/ --ignore=tests/eval -v
```

| Layer | What it tests | Method |
|:------|:-------------|:-------|
| **Agent Quality** | Faithfulness, completeness, reasonableness of each agent's output | LLM-as-judge (real API calls → Gemini scores 1-5) |
| **Truth Checker** | Conflict detection accuracy on known scenarios | Mock reports + real LLM cross-validation |
| **E2E Regression** | Full pipeline on representative tickers (AAPL, TSLA) | Pipeline completion + LLM-as-judge on final report |

---

## Key Technical Decisions

| Decision | Rationale |
|:---------|:----------|
| **LangGraph** over CrewAI/AutoGen | Native support for cyclic graphs, conditional routing, and parallel fan-out/fan-in |
| **Gemini 2.5 Flash** | 1M free tokens/day, strong structured output support |
| **Grok `x_search`** | Only API with native access to real-time X/Twitter data |
| **Pydantic structured output** | Type-safe agent communication, no parsing failures |
| **`graph.stream()`** in UI | Enables per-node UI updates without threading issues |
| **RSI-14 (Wilder method)** | Industry-standard momentum indicator, not just simple moving average |
| **20-F support** | Handles foreign-listed companies (ADRs like BABA, TSM) |

---

## How the Graph Routing Works

1. **Supervisor** resolves user input to a ticker via LLM. If unresolvable → graph exits gracefully.
2. **Fan-out**: Sentiment Scout, Market Quant, and SEC Auditor execute in parallel.
3. **Fan-in**: All three reports converge at Truth Checker.
4. **Truth Checker** evaluates consistency:
   - `consistent` / `minor_conflicts` → proceed to Report Generator
   - `needs_more_data` (round 1 only) → loop back to Supervisor for round 2
5. **Report Generator** synthesizes a calibrated 1–10 risk assessment.
6. Market Quant skips re-computation on round 2 (yfinance data doesn't change in minutes).

<img width="996" height="627" alt="Screenshot 2026-02-28 at 19 34 51" src="https://github.com/user-attachments/assets/e20b35d9-f5e6-4f72-af0d-0fb0cdeca3b6" />

---

## License

MIT

---

<div align="center">

**Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [Gemini](https://ai.google.dev/) · [Grok](https://x.ai/) · [Tavily](https://tavily.com/)**

</div>
