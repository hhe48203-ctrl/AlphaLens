import yfinance as yf
import numpy as np
from langchain_core.messages import AIMessage
from app.state import AlphaLensState, MarketQuantReport
from app.config import get_llm

TAG = "📊 [Market Quant]"


def _fetch_stock_data(ticker: str) -> dict:
    """Fetch 3 months of stock data via yfinance and compute key indicators"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")

    if hist.empty:
        raise ValueError(f"Could not retrieve stock data for {ticker}")

    closes = hist["Close"]
    volumes = hist["Volume"]

    # Daily returns
    returns = closes.pct_change().dropna()

    # Annualized volatility = daily volatility * sqrt(252)
    volatility = float(returns.std() * np.sqrt(252))

    # Sharpe ratio (assuming risk-free rate of 4.5%)
    risk_free_rate = 0.045
    annualized_return = float(returns.mean() * 252)
    sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

    # 30-day price change
    if len(closes) >= 22:
        price_change_30d = float((closes.iloc[-1] / closes.iloc[-22] - 1) * 100)
    else:
        price_change_30d = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)

    # Volume trend: compare 5-day avg vs 20-day avg
    recent_vol = float(volumes.tail(5).mean())
    older_vol = float(volumes.tail(20).mean())
    if recent_vol > older_vol * 1.1:
        vol_trend = "increasing"
    elif recent_vol < older_vol * 0.9:
        vol_trend = "decreasing"
    else:
        vol_trend = "stable"

    # RSI-14 (Wilder smoothing method)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = round(float(rsi.iloc[-1]), 2)

    return {
        "current_price": round(float(closes.iloc[-1]), 2),
        "price_change_30d_pct": round(price_change_30d, 2),
        "volatility_30d": round(volatility, 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "rsi_14": rsi_value,
        "high_52w": round(float(closes.max()), 2),
        "low_52w": round(float(closes.min()), 2),
        "avg_volume": int(volumes.mean()),
        "volume_trend": vol_trend,
        "data_points": len(closes),
    }


def market_quant_node(state: AlphaLensState) -> dict:
    ticker = state["ticker"]
    print(f"{TAG} Computing quantitative indicators for {ticker}...")

    # If previous round already has a report, reuse it (yfinance data won't change in minutes)
    existing = state.get("market_quant_report")
    if existing:
        print(f"{TAG} ⏩ Reusing previous round data (Price: ${existing.current_price} | Signal: {existing.technical_signal})")
        return {}

    try:
        print(f"{TAG} Fetching 3-month data via yfinance...")
        data = _fetch_stock_data(ticker)
    except Exception as e:
        print(f"{TAG} ❌ Data fetch failed: {e}")
        return {"errors": [f"Market Quant failed: {str(e)}"]}

    print(f"{TAG} Retrieved {data['data_points']} trading days")
    print(f"{TAG} Price: ${data['current_price']} | 30-Day Change: {data['price_change_30d_pct']}%")
    print(f"{TAG} Volatility: {data['volatility_30d']} | Sharpe: {data['sharpe_ratio']} | RSI-14: {data['rsi_14']}")
    print(f"{TAG} Volume Trend: {data['volume_trend']} | Avg Volume: {data['avg_volume']:,}")

    # LLM generates technical analysis
    print(f"{TAG} Calling LLM for technical analysis...")
    llm = get_llm()
    prompt = f"""You are a quantitative analyst. Based on the following data for {ticker}, provide a brief technical analysis summary (3-5 sentences) and clearly state whether the current technical signal is bullish, bearish, or neutral.

Data:
- Current Price: ${data['current_price']}
- 30-Day Change: {data['price_change_30d_pct']}%
- Annualized Volatility: {data['volatility_30d']}
- Sharpe Ratio: {data['sharpe_ratio']}
- RSI-14: {data['rsi_14']} (below 30 = oversold, above 70 = overbought)
- Volume Trend: {data['volume_trend']}

Respond in English only. Be concise and direct. The last line must be: Technical Signal: bullish / bearish / neutral"""

    response = llm.invoke(prompt)
    summary = response.content

    # Extract signal from LLM response
    signal = "neutral"
    summary_lower = summary.lower()
    if "bullish" in summary_lower:
        signal = "bullish"
    elif "bearish" in summary_lower:
        signal = "bearish"

    report = MarketQuantReport(
        ticker=ticker,
        current_price=data["current_price"],
        price_change_30d_pct=data["price_change_30d_pct"],
        volatility_30d=data["volatility_30d"],
        sharpe_ratio=data["sharpe_ratio"],
        rsi_14=data["rsi_14"],
        technical_signal=signal,
        raw_data_summary=summary,
        confidence=0.85,
    )

    print(f"{TAG} ✅ Done | Price: ${report.current_price} | Signal: {report.technical_signal}")

    return {
        "market_quant_report": report,
        "messages": [AIMessage(content=f"[Market Quant] {summary}", name="market_quant")],
    }
