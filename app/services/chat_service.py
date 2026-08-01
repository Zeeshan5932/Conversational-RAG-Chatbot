from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from app.agents.graph import agent_graph
from app.utils.logger import logger


class ChatService:
    """Service wrapper for executing conversations against state graph threads."""

    async def process_chat_message(self, message: str, thread_id: str = "default") -> Dict[str, Any]:
        """Invokes compiled LangGraph agent state engine with thread configuration."""
        logger.info(f"Processing chat message (Thread: {thread_id}): '{message[:50]}...'")

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "query": message,
            "user_query": message,
            "route": "",
            "route_decision": "",
            "context": {},
            "citations": [],
            "thread_id": thread_id,
        }

        # Session configuration for thread persistence
        config = {"configurable": {"thread_id": thread_id}}

        # Run compiled state graph
        output_state = agent_graph.invoke(initial_state, config=config)

        # Merge citations from web search, RAG, or URL research
        citations = output_state.get("citations", [])
        if not citations and output_state.get("retrieved_docs"):
            citations = [
                {"url": "", "source": d.get("source", ""), "page": d.get("page")}
                for d in output_state.get("retrieved_docs", [])
            ]

        context = output_state.get("context", {}) or {}
        docs_retrieved: List[Dict[str, Any]] = output_state.get("retrieved_docs", []) or context.get("documents", []) or []

        return {
            "answer": output_state.get("final_answer", "No answer generated."),
            "route_used": output_state.get("route") or output_state.get("route_decision", "unknown"),
            "docs_retrieved": docs_retrieved,
            "search_results": output_state.get("search_results", []) or context.get("web_results", []),
            "citations": citations
        }