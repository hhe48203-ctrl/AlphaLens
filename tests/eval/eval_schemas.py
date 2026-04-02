"""Pydantic schemas for LLM-as-judge evaluation results."""

from pydantic import BaseModel, Field


class AgentEvalResult(BaseModel):
    """LLM judge scores for an individual agent's output quality."""
    faithfulness: int = Field(ge=1, le=5)
    faithfulness_reasoning: str
    completeness: int = Field(ge=1, le=5)
    completeness_reasoning: str
    reasonableness: int = Field(ge=1, le=5)
    reasonableness_reasoning: str


class ReportEvalResult(BaseModel):
    """LLM judge scores for the final pipeline report."""
    coherence: int = Field(ge=1, le=5)
    coherence_reasoning: str
    balance: int = Field(ge=1, le=5)
    balance_reasoning: str
    calibration: int = Field(ge=1, le=5)
    calibration_reasoning: str
