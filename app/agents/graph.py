"""LangGraph assembly for the research assistant."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes import general_llm_node, rag_node, synthesis_node, url_reader_node, web_search_node
from app.agents.router import router_node
from app.agents.state import AgentState
from app.utils.logger import logger
from app.memory.conversation import memory_checkpointer


def route_decision_edge(state: AgentState) -> str:
    """Map router output into a valid graph target."""
    route = state.get("route") or state.get("route_decision") or "general"
    logger.info("LangGraph route selected: %s", route)

    valid_routes = {"general", "rag", "web_search", "url_reader"}
    return route if route in valid_routes else "general"


def build_graph() -> StateGraph:
    """Construct and compile the state graph with thread persistence."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("url_reader", url_reader_node)
    workflow.add_node("general", general_llm_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_decision_edge,
        {
            "general": "general",
            "rag": "rag",
            "web_search": "web_search",
            "url_reader": "url_reader",
        },
    )

    workflow.add_edge("rag", "synthesis")
    workflow.add_edge("web_search", "synthesis")
    workflow.add_edge("url_reader", "synthesis")
    workflow.add_edge("general", END)
    workflow.add_edge("synthesis", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


agent_graph = build_graph()