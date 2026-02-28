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

    # Fan-in: all 3 Agents complete -> Truth Checker
    builder.add_edge("sentiment_scout", "truth_checker")
    builder.add_edge("sec_auditor", "truth_checker")
    builder.add_edge("market_quant", "truth_checker")

    # Conditional routing: Truth Checker decides loop vs finish
    builder.add_conditional_edges(
        "truth_checker",
        route_after_truth_check,
        {"supervisor": "supervisor", "report_generator": "report_generator"},
    )

    builder.add_edge("report_generator", END)

    return builder.compile()
