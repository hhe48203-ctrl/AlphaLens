import os
from openai import OpenAI
from tavily import TavilyClient
from langchain_core.messages import AIMessage
from app.state import AlphaLensState, SentimentReport
from app.config import get_llm

TAG = "🔍 [Sentiment Scout]"


def _search_x(ticker: str) -> str:
    """Use Grok x_search to find real-time discussions about the stock on X"""
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )

    response = client.responses.create(
        model="grok-4-fast-non-reasoning",
        tools=[{"type": "x_search"}],
        input=f"What is the latest sentiment about ${ticker} stock on X? "
              f"Summarize the key narratives, notable accounts, and overall mood. "
              f"Include direct quotes from posts.",
        max_output_tokens=800,
    )

    # Extract text content from response
    texts = []
    for item in response.output:
        if hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
        elif hasattr(item, "text"):
            texts.append(item.text)

    return "\n".join(texts) if texts else "No X platform data retrieved"


def _search_news(ticker: str) -> str:
    """Use Tavily to search related news articles"""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    results = client.search(
        query=f"{ticker} stock news analysis",
        topic="finance",
        max_results=5,
        time_range="week",
    )

    # Concatenate search results
    summaries = []
    for r in results.get("results", []):
        summaries.append(f"• {r.get('title', '')}: {r.get('content', '')[:200]}")

    return "\n".join(summaries) if summaries else "No news data retrieved"


def sentiment_node(state: AlphaLensState) -> dict:
    ticker = state["ticker"]
    print(f"{TAG} Starting real-time sentiment search for {ticker}...")

    # Step 1: Search X platform
    try:
        print(f"{TAG} Calling Grok x_search for X platform...")
        x_data = _search_x(ticker)
        x_len = len(x_data)
        print(f"{TAG} X platform data retrieved ({x_len} chars)")
    except Exception as e:
        print(f"{TAG} ⚠️ X search failed: {e}")
        x_data = f"X search failed: {str(e)}"

    # Step 2: Search news
    try:
        print(f"{TAG} Calling Tavily news search...")
        news_data = _search_news(ticker)
        news_count = news_data.count("•")
        print(f"{TAG} Retrieved {news_count} news results")
    except Exception as e:
        print(f"{TAG} ⚠️ News search failed: {e}")
        news_data = f"News search failed: {str(e)}"

    # Step 3: LLM sentiment analysis
    print(f"{TAG} Calling LLM for sentiment analysis...")
    llm = get_llm()
    structured_llm = llm.with_structured_output(SentimentReport)

    prompt = f"""You are a sentiment analyst. Based on the following data collected from X (Twitter) and news sources, analyze the market sentiment for {ticker}. Respond in English only.

=== X Platform Discussions ===
{x_data}

=== News Coverage ===
{news_data}

Fill in the following fields:
- ticker: "{ticker}"
- sentiment_score: from -1.0 (extremely bearish) to 1.0 (extremely bullish)
- volume_change_pct: estimated discussion volume change percentage
- key_narratives: extract 3-5 dominant narratives
- raw_evidence: list 3-5 most representative direct quotes or sources
- confidence: your confidence in this analysis (0-1)"""

    try:
        report = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"{TAG} ❌ LLM analysis failed: {e}")
        return {"errors": [f"Sentiment Scout LLM failed: {str(e)}"]}

    # Build natural language message
    summary = (
        f"Sentiment Score: {report.sentiment_score}, "
        f"Volume Change: {report.volume_change_pct}%, "
        f"Key Narratives: {', '.join(report.key_narratives[:3])}"
    )

    mood = "Bullish 📈" if report.sentiment_score > 0.3 else ("Bearish 📉" if report.sentiment_score < -0.3 else "Neutral ➡️")
    print(f"{TAG} ✅ Done | Mood: {mood} ({report.sentiment_score}) | Volume Change: {report.volume_change_pct}% | Confidence: {report.confidence}")

    return {
        "sentiment_report": report,
        "messages": [AIMessage(content=f"[Sentiment Scout] {summary}", name="sentiment_scout")],
    }
