from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes import (
    router_node,
    general_llm_node,
    rag_node,
    web_search_node,
    url_research_node,
    hybrid_node
)
from app.utils.logger import logger


def route_decision_edge(state: AgentState) -> str:
    """Conditional edge routing logic mapping decisions to state graph execution targets."""
    route = state.get("route_decision", "general_llm")
    logger.info(f"LangGraph Edge Evaluated Route -> {route}")
    
    valid_routes = {"general_llm", "rag", "web_search", "url_research", "hybrid"}
    return route if route in valid_routes else "general_llm"


def build_graph() -> StateGraph:
    """Constructs and compiles the state graph with in-memory persistence checkpointing."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("general_llm", general_llm_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("url_research", url_research_node)
    workflow.add_node("hybrid", hybrid_node)

    # Entry point
    workflow.set_entry_point("router")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "router",
        route_decision_edge,
        {
            "general_llm": "general_llm",
            "rag": "rag",
            "web_search": "web_search",
            "url_research": "url_research",
            "hybrid": "hybrid"
        }
    )

    # Connect Execution Nodes to END
    workflow.add_edge("general_llm", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("web_search", END)
    workflow.add_edge("url_research", END)
    workflow.add_edge("hybrid", END)

    # Checkpointer for stateful multi-turn session persistence
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Single compiled graph application instance
agent_graph = build_graph()