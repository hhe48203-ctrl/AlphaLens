from langgraph.graph import StateGraph, END, START

from app.state import AlphaLensState
from app.agents.supervisor import supervisor_node
from app.agents.sentiment import sentiment_node
from app.agents.sec_auditor import sec_auditor_node
from app.agents.market_quant import market_quant_node
from app.agents.truth_checker import truth_checker_node
from app.agents.reporter import report_generator_node


def route_after_supervisor(state: AlphaLensState) -> list[str]:
    """
    Routing after Supervisor:
    - If status == "completed" (could not resolve ticker), go to END
    - Otherwise, fan-out to 3 data Agents
    """
    if state.get("status") == "completed":
        return ["__end__"]
    return ["sentiment_scout", "sec_auditor", "market_quant"]


def data_quality_gate_node(state: AlphaLensState) -> dict:
    """
    Gate before Truth Checker:
    - Hard-fail when all data agents failed (0/3 reports available)
    - Continue with warning when partial data is available
    """
    report_count = sum(
        bool(state.get(key))
        for key in ("sentiment_report", "sec_report", "market_quant_report")
    )

    if report_count == 0:
        error_msg = "All data agents failed. No reliable inputs available for truth checking."
        print(f"❌ Data Quality Gate: {error_msg}")
        return {"status": "error", "errors": [error_msg]}

    if report_count < 3:
        missing = 3 - report_count
        warning_msg = (
            f"Proceeding with incomplete evidence: {report_count}/3 reports available, "
            f"{missing} agent(s) missing."
        )
        print(f"⚠️ Data Quality Gate: {warning_msg}")
        return {"errors": [warning_msg]}

    print("✅ Data Quality Gate: All 3 agent reports available")
    return {}


def route_after_data_quality_gate(state: AlphaLensState) -> str:
    """Route based on data quality result."""
    if state.get("status") == "error":
        return "__end__"
    return "truth_checker"


def route_after_truth_check(state: AlphaLensState) -> str:
    """
    Routing after Truth Checker:
    - If major conflicts found AND under max iterations -> loop back to Supervisor
    - Otherwise -> generate final report
    """
    report = state.get("truth_check_report")
    if not report:
        return "report_generator"

    if report.recommendation == "needs_more_data" and state["iteration_count"] < state["max_iterations"]:
        return "supervisor"

    return "report_generator"


def build_graph():
    builder = StateGraph(AlphaLensState)

    # Register all nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sentiment_scout", sentiment_node)
    builder.add_node("sec_auditor", sec_auditor_node)
    builder.add_node("market_quant", market_quant_node)
    builder.add_node("data_quality_gate", data_quality_gate_node)
    builder.add_node("truth_checker", truth_checker_node)
    builder.add_node("report_generator", report_generator_node)

    # Entry -> Supervisor
    builder.add_edge(START, "supervisor")

    # Conditional routing: Supervisor -> Fan-out or END
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        ["sentiment_scout", "sec_auditor", "market_quant", "__end__"],
    )

    # Fan-in: all 3 Agents complete -> Data quality gate
    builder.add_edge("sentiment_scout", "data_quality_gate")
    builder.add_edge("sec_auditor", "data_quality_gate")
    builder.add_edge("market_quant", "data_quality_gate")

    # Data quality gate: continue to Truth Checker or end on hard failure
    builder.add_conditional_edges(
        "data_quality_gate",
        route_after_data_quality_gate,
        {"truth_checker": "truth_checker", "__end__": END},
    )

    # Conditional routing: Truth Checker decides loop vs finish
    builder.add_conditional_edges(
        "truth_checker",
        route_after_truth_check,
        {"supervisor": "supervisor", "report_generator": "report_generator"},
    )

    builder.add_edge("report_generator", END)

    return builder.compile()
