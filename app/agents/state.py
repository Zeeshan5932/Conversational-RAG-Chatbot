"""Shared LangGraph state definitions."""

from __future__ import annotations

import operator
from typing import Any, Annotated, Dict, List, NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """Shared state for the research assistant graph."""

    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    route: str
    context: Dict[str, Any]
    citations: Annotated[List[Dict[str, Any]], operator.add]
    thread_id: str

    # Backward-compatible fields used by existing service code.
    user_query: NotRequired[str]
    route_decision: NotRequired[str]
    final_answer: NotRequired[str]
    retrieved_docs: NotRequired[List[Dict[str, Any]]]
    search_results: NotRequired[List[Dict[str, Any]]]
    url_content: NotRequired[str]