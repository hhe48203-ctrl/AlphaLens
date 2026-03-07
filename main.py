from dotenv import load_dotenv

# Load .env file first, ensuring API keys are injected before any module imports
load_dotenv()

# LangSmith tracing (requires LANGCHAIN_API_KEY in .env)
import os
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "AlphaLens")

from app.graph import build_graph
from app.reporter import save_detailed_report


def main():
    # Build and compile the LangGraph state graph
    graph = build_graph()

    # Get user input (supports natural language, e.g. "NVDA", "google", "pick a defense stock")
    user_input = input("Enter stock to investigate (ticker, company name, or description): ").strip()
    if not user_input:
        print("No input provided, exiting.")
        return

    # Initial state (ticker left empty, Supervisor will resolve via LLM)
    initial_state = {
        "user_query": user_input,
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

    print("=" * 60)
    print("  AlphaLens — Multi-Agent Financial Intelligence")
    print("=" * 60)
    print()

    # Invoke graph synchronously; LangGraph handles parallelism and loops
    result = graph.invoke(initial_state)

    # Graph may exit early due to unresolved ticker or upstream data failures
    if not result.get("final_report"):
        print()
        print("=" * 60)
        if result.get("status") == "error":
            print("  Analysis failed (insufficient data from agents)")
            for err in result.get("errors", []):
                print(f"  - {err}")
        else:
            print("  Analysis not performed (could not identify target stock)")
        print("=" * 60)
        return

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Status: {result['status']}")
    print(f"  Total iterations: {result['iteration_count']}")

    report = result["final_report"]
    level = report.risk_level
    risk_labels = {1: "Very Low", 2: "Very Low", 3: "Low", 4: "Low", 5: "Neutral",
                   6: "Elevated", 7: "High", 8: "High", 9: "Very High", 10: "Critical"}
    bar = "█" * level + "░" * (10 - level)
    print(f"  Risk Level: [{bar}] {level}/10 ({risk_labels.get(level, 'Unknown')})")
    print(f"  Summary: {report.executive_summary}")

    # Save detailed Markdown report
    save_detailed_report(result)


if __name__ == "__main__":
    main()
