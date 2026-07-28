from typing import Dict, Any
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
            "user_query": message,
            "route_decision": "",
            "retrieved_docs": [],
            "search_results": [],
            "url_content": "",
            "final_answer": "",
            "citations": []
        }

        # Session configuration for thread persistence
        config = {"configurable": {"thread_id": thread_id}}

        # Run compiled state graph
        output_state = agent_graph.invoke(initial_state, config=config)

        # Merge citations from web search, RAG, or URL research
        citations = output_state.get("citations", [])
        if not citations and output_state.get("retrieved_docs"):
            citations = [
                {"source": d.get("source"), "page": d.get("page")}
                for d in output_state.get("retrieved_docs", [])
            ]

        return {
            "answer": output_state.get("final_answer", "No answer generated."),
            "route_used": output_state.get("route_decision", "unknown"),
            "docs_retrieved": output_state.get("retrieved_docs", []),
            "search_results": output_state.get("search_results", []),
            "citations": citations
        }