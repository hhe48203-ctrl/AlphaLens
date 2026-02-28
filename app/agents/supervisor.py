from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from app.state import AlphaLensState
from app.config import get_llm


class TickerResolution(BaseModel):
    """LLM output for resolving user input to a ticker"""
    ticker: str = Field(description="Resolved US stock ticker symbol, e.g. AAPL, GOOGL, NVDA. Empty string if cannot resolve.")
    company_name: str = Field(description="Full company name in English")
    reasoning: str = Field(description="Brief explanation of how you resolved this")
    resolved: bool = Field(description="True if successfully resolved to a specific stock, False otherwise")


def _resolve_ticker(user_query: str) -> TickerResolution:
    """Use LLM to resolve natural language input to a stock ticker"""
    llm = get_llm()
    structured_llm = llm.with_structured_output(TickerResolution)

    prompt = f"""You are a financial assistant. The user wants to investigate a stock. Based on their input, determine which US-listed stock they are referring to and return its ticker symbol.

User input: "{user_query}"

Rules:
- If the user gives a ticker directly (e.g. "AAPL", "tsla"), use it as-is (uppercase).
- If the user gives a company name (e.g. "google", "nvidia", "apple"), resolve it to the correct US ticker (GOOGL, NVDA, AAPL).
- If the user gives a description (e.g. "the biggest EV company", "largest smartphone maker"), infer the most likely company and return its ticker.
- If the user asks for a random/arbitrary stock (e.g. "pick a random tech stock", "any defense stock"), pick ONE specific well-known stock that fits and return its ticker.
- If the input is completely unrelated to stocks/investing (e.g. "what's the weather", "hello"), set resolved=False and ticker="".
- Always prefer the primary US listing ticker (e.g. GOOGL not GOOG, META not FB)."""

    return structured_llm.invoke(prompt)


def supervisor_node(state: AlphaLensState) -> dict:
    """
    Supervisor: handles task dispatch and flow control.
    - Round 0: resolve user input -> ticker, initial dispatch
    - Subsequent rounds: conflict-driven re-investigation
    """
    iteration = state["iteration_count"]

    if iteration == 0:
        user_query = state["user_query"]
        print(f"🧠 Supervisor: Received query -> {user_query}")

        # Resolve user input via LLM
        print("   🔍 Resolving target stock...")
        try:
            resolution = _resolve_ticker(user_query)
        except Exception as e:
            print(f"   ❌ Resolution failed: {e}")
            return {"status": "completed"}

        if not resolution.resolved or not resolution.ticker:
            print(f"   ❌ Could not identify target stock, exiting")
            print(f"   💬 Reason: {resolution.reasoning}")
            return {"status": "completed"}

        ticker = resolution.ticker.upper()
        print(f"   ✅ Resolved: {resolution.company_name} ({ticker})")
        print(f"   💬 {resolution.reasoning}")
        print("   Dispatching tasks to 3 Agents (parallel execution)...")

        msg = f"[Supervisor] Starting analysis for {resolution.company_name} ({ticker}), dispatching to Sentiment Scout / SEC Auditor / Market Quant"
        return {
            "ticker": ticker,
            "messages": [AIMessage(content=msg, name="supervisor")],
        }

    else:
        ticker = state["ticker"]
        truth = state.get("truth_check_report")

        if truth and truth.conflicts:
            conflict_summary = "; ".join([c.explanation for c in truth.conflicts[:2]])
            reason_label = "conflict-driven"
            reason_detail = conflict_summary[:120]
        else:
            reason_label = "data quality / low confidence"
            reason_detail = truth.summary[:120] if truth else "insufficient data"

        print(f"🧠 Supervisor: Round {iteration + 1} investigation ({reason_label})")
        print(f"   Reason: {reason_detail}")
        print("   Re-dispatching tasks to 3 Agents...")
        msg = f"[Supervisor] Round {iteration + 1} re-investigation ({reason_label}): {reason_detail}"

        return {
            "messages": [AIMessage(content=msg, name="supervisor")],
        }
