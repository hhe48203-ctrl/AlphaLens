import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# -- Agent output schemas (Pydantic structured output) --

class SentimentReport(BaseModel):
    """Sentiment Scout Agent output: X platform and news sentiment analysis"""
    ticker: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)  # -1 extremely bearish, 1 extremely bullish
    volume_change_pct: float                          # Discussion volume change vs previous day (%)
    key_narratives: list[str]                         # Dominant narratives (max 5)
    raw_evidence: list[str]                           # Raw evidence (quotes or links)
    confidence: float = Field(ge=0, le=1)             # Confidence score


class SECReport(BaseModel):
    """SEC Auditor Agent output: key filing information extraction"""
    ticker: str
    filing_type: str                                  # Filing type, e.g. "10-K", "20-F"
    risk_factors: list[str]                           # Major risk factors from filings
    key_financial_metrics: dict[str, float]           # Key metrics, e.g. PE, debt-to-equity
    red_flags: list[str]                              # Warning signals
    raw_evidence: list[str]
    confidence: float = Field(ge=0, le=1)


class MarketQuantReport(BaseModel):
    """Market Quant Agent output: price action and technical indicators"""
    ticker: str
    current_price: float
    price_change_30d_pct: float                       # 30-day price change (%)
    volatility_30d: float                             # 30-day annualized volatility
    sharpe_ratio: float                               # Sharpe ratio (risk-adjusted return)
    rsi_14: float                                     # 14-day RSI indicator
    technical_signal: Literal["bullish", "bearish", "neutral"]
    raw_data_summary: str
    confidence: float = Field(ge=0, le=1)


class Conflict(BaseModel):
    """Single conflict record between two Agents"""
    source_a: str                                     # Source Agent A
    claim_a: str                                      # A's conclusion
    source_b: str                                     # Source Agent B
    claim_b: str                                      # B's contradicting conclusion
    severity: Literal["low", "medium", "high", "critical"]
    explanation: str                                  # Explanation of the conflict


class TruthCheckReport(BaseModel):
    """Truth Checker Agent output: cross-validation results and recommendation"""
    conflicts: list[Conflict]
    overall_consistency: float = Field(ge=0, le=1)   # Overall consistency score
    recommendation: Literal["consistent", "minor_conflicts", "major_conflicts", "needs_more_data"]
    summary: str


class FinalReport(BaseModel):
    """Final investment risk report"""
    ticker: str
    risk_level: int = Field(ge=1, le=10)              # Risk level 1-10 (1 lowest, 10 highest)
    executive_summary: str                            # Executive summary
    sentiment_section: str
    fundamentals_section: str
    technical_section: str
    conflicts_section: str                            # Conflicts and uncertainty section


# -- LangGraph shared state --

class AlphaLensState(TypedDict):
    # User input
    user_query: str
    ticker: str
    messages: Annotated[list[BaseMessage], add_messages]  # LangGraph built-in message list

    # Agent reports (all initially None)
    sentiment_report: SentimentReport | None
    sec_report: SECReport | None
    market_quant_report: MarketQuantReport | None
    truth_check_report: TruthCheckReport | None
    final_report: FinalReport | None

    # Human-in-the-loop report controls
    risk_bias: Literal["conservative", "balanced", "aggressive"]
    report_language: Literal["en", "zh"]
    reporter_guidance: str

    # Flow control fields
    iteration_count: int    # Current iteration count
    max_iterations: int     # Max allowed iterations (prevents infinite loops)
    status: Literal["in_progress", "completed", "error"]
    errors: Annotated[list[str], operator.add]
