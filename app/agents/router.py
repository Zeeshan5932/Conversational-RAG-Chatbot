"""Intent router for the research assistant graph."""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.llm.groq import get_llm
from app.utils.logger import logger


class QueryRoute(BaseModel):
    """Structured routing decision returned by the LLM."""

    route: Literal["rag", "web_search", "url_reader", "general"] = Field(
        description="The route selected for the user query.",
    )
    reasoning: str = Field(
        description="Short justification for the selected route.",
    )


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an intent classifier for a research assistant.
Classify the user's query into exactly one route:

- rag: Questions about uploaded documents, internal files, PDFs, or stored knowledge base content.
- web_search: Questions requiring current, recent, or live web information.
- url_reader: Requests that explicitly mention a URL or ask to read, summarize, or analyze a webpage.
- general: Greetings, small talk, reasoning, coding help, or questions that do not require retrieval.

Return only the best route and a short reason.""",
        ),
        ("human", "Query: {query}"),
    ]
)


def route_query(query: str) -> QueryRoute:
    """Classify a query into one of the supported execution routes."""
    logger.info("Routing query: %s", query)

    try:
        structured_llm = get_llm().with_structured_output(QueryRoute)
        chain = ROUTER_PROMPT | structured_llm
        result: QueryRoute = chain.invoke({"query": query})
        logger.info("Route selected: %s", result.route)
        return result
    except Exception as exc:
        logger.exception("Routing failed, defaulting to general: %s", exc)
        return QueryRoute(route="general", reasoning="Fallback after router failure.")


def router_node(state: AgentState) -> dict:
    """LangGraph node that stores the routing decision in state."""
    query = state.get("query") or state.get("user_query") or ""
    decision = route_query(query)

    return {
        "query": query,
        "route": decision.route,
        "route_decision": decision.route,
        "context": {**state.get("context", {}), "router_reasoning": decision.reasoning},
    }