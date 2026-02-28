import os
import requests
from tavily import TavilyClient
from langchain_core.messages import AIMessage
from app.state import AlphaLensState, SECReport
from app.config import get_llm

TAG = "⚖️ [SEC Auditor]"

# SEC EDGAR requires User-Agent identification in request headers
SEC_HEADERS = {
    "User-Agent": "AlphaLens Research contact@alphalens.dev",
    "Accept": "application/json",
}


def _get_cik(ticker: str) -> str:
    """Convert stock ticker to CIK number via SEC EDGAR"""
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(tickers_url, headers=SEC_HEADERS)
    resp.raise_for_status()
    data = resp.json()

    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            # CIK must be zero-padded to 10 digits
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Could not find CIK for {ticker}")


def _fetch_sec_filings(ticker: str) -> tuple[str, dict]:
    """Fetch recent 10-K or 20-F filings from SEC EDGAR"""
    cik = _get_cik(ticker)

    # Fetch the company's recent filing list
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    data = resp.json()

    # Search for 10-K or 20-F (foreign companies use 20-F)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    target_forms = ["10-K", "20-F"]
    filing_info = []
    found_type = None
    latest_date = "N/A"

    for i, form in enumerate(forms):
        if form in target_forms and i < len(dates):
            filing_info.append(f"Type: {form}, Date: {dates[i]}, Accession: {accessions[i]}")
            if found_type is None:
                found_type = form
                latest_date = dates[i]
            if len(filing_info) >= 3:
                break

    # Use Tavily to search for supplementary filing analysis
    filing_keyword = found_type or "10-K 20-F"
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        search_results = tavily.search(
            query=f"{ticker} SEC {filing_keyword} filing risk factors financial analysis",
            topic="finance",
            max_results=5,
            time_range="year",
        )
        news_snippets = []
        for r in search_results.get("results", []):
            news_snippets.append(f"• {r.get('title', '')}: {r.get('content', '')[:300]}")
        news_context = "\n".join(news_snippets)
        news_count = len(news_snippets)
    except Exception:
        news_context = "Tavily search failed"
        news_count = 0

    company_name = data.get("name", ticker)
    sic = data.get("sic", "N/A")
    sic_desc = data.get("sicDescription", "Unknown")

    filing_label = found_type or "N/A"
    result = f"""Company: {company_name}
Industry: {sic_desc} (SIC: {sic})
Recent {filing_label} Filings:
{chr(10).join(filing_info) if filing_info else "No 10-K or 20-F filings found"}

Related Analysis:
{news_context}"""

    meta = {
        "cik": cik,
        "company_name": company_name,
        "industry": sic_desc,
        "filing_type": filing_label,
        "filing_count": len(filing_info),
        "latest_date": latest_date,
        "news_count": news_count,
    }

    return result, meta


def sec_auditor_node(state: AlphaLensState) -> dict:
    ticker = state["ticker"]
    print(f"{TAG} Fetching SEC filings for {ticker}...")

    # Step 1: Fetch SEC data
    try:
        print(f"{TAG} Querying SEC EDGAR (CIK lookup + filing list)...")
        sec_data, meta = _fetch_sec_filings(ticker)
    except Exception as e:
        print(f"{TAG} ❌ SEC data fetch failed: {e}")
        return {"errors": [f"SEC Auditor failed: {str(e)}"]}

    print(f"{TAG} Company: {meta['company_name']} | Industry: {meta['industry']} | CIK: {meta['cik']}")
    print(f"{TAG} Found {meta['filing_count']} {meta['filing_type']} filings | Latest: {meta['latest_date']}")
    print(f"{TAG} Tavily supplemented {meta['news_count']} analysis articles")

    # Step 2: LLM analysis
    print(f"{TAG} Calling LLM to extract risk factors and key metrics...")
    llm = get_llm()
    structured_llm = llm.with_structured_output(SECReport)

    prompt = f"""You are an SEC filing analysis expert. Based on the following SEC data and related analysis for {ticker}, extract key information. Respond in English only.

{sec_data}

Note: Foreign private issuers (e.g. Chinese ADRs) file 20-F instead of 10-K. Both serve the same purpose as annual reports.

Fill in the following fields:
- ticker: "{ticker}"
- filing_type: the primary filing type found (e.g. "10-K" or "20-F")
- risk_factors: list 3-5 major risk factors
- key_financial_metrics: extract available financial metrics (e.g. PE, debt-to-equity) in dict format
- red_flags: list any warning signals (insider trading, accounting adjustments, etc.)
- raw_evidence: list original data sources
- confidence: your confidence in this analysis (0-1)

If certain information is unavailable, note "limited data" in risk_factors."""

    try:
        report = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"{TAG} ❌ LLM analysis failed: {e}")
        return {"errors": [f"SEC Auditor LLM failed: {str(e)}"]}

    summary = f"Filing: {report.filing_type}, Risk Factors: {len(report.risk_factors)}, Red Flags: {len(report.red_flags)}"
    red_flag_text = f" | 🚩 {', '.join(report.red_flags[:2])}" if report.red_flags else ""
    print(f"{TAG} ✅ Done | {summary} | Confidence: {report.confidence}{red_flag_text}")

    return {
        "sec_report": report,
        "messages": [AIMessage(content=f"[SEC Auditor] {summary}", name="sec_auditor")],
    }
