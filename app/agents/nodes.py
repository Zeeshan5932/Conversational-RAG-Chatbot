"""Execution nodes for the research assistant graph."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.llm.groq import get_llm
from app.rag.retriever import DocumentRetriever
from app.rag.vectorstore import VectorStoreManager
from app.tools.url_reader import URLReaderTool, extract_url_from_text
from app.tools.web_search import WebSearchTool
from app.utils.citations import build_citation, dedupe_citations
from app.utils.logger import logger


web_search_tool = WebSearchTool(max_results=5)
url_reader_tool = URLReaderTool()


def _query_from_state(state: AgentState) -> str:
    return (state.get("query") or state.get("user_query") or "").strip()


def _append_context(state: AgentState, **updates: Any) -> Dict[str, Any]:
    context = dict(state.get("context", {}))
    context.update(updates)
    return context


def _error_message(message: str, route: str) -> Dict[str, Any]:
    return {
        "messages": [AIMessage(content=message)],
        "final_answer": message,
        "route": route,
        "route_decision": route,
        "context": {"error": message},
    }


def rag_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve internal documents and prepare grounded context for synthesis."""
    logger.info("Executing rag_node")
    query = _query_from_state(state)

    try:
        vector_store_mgr = VectorStoreManager()
        retriever = DocumentRetriever(vector_store_mgr, k=4)
        docs = retriever.retrieve(query)
    except Exception as exc:
        logger.exception("RAG retrieval failed: %s", exc)
        return _error_message(
            "I could not access the document index right now. Please check the embedding and ChromaDB configuration.",
            "rag",
        )

    formatted_context: List[str] = []
    retrieved_docs: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        formatted_context.append(
            f"[Document {index}] Source: {source} | Page: {page if page is not None else 'n/a'}\n{doc.page_content}"
        )
        retrieved_docs.append(
            {
                "source": source,
                "page": page,
                "content": doc.page_content[:500],
            }
        )
        citations.append(build_citation(url="", source=source, page=page))

    context_text = "\n\n".join(formatted_context)
    context = _append_context(
        state,
        mode="rag",
        documents=[
            {
                "source": item["source"],
                "page": item["page"],
                "content": item["content"],
            }
            for item in retrieved_docs
        ],
        context_text=context_text,
    )

    return {
        "query": query,
        "route": "rag",
        "route_decision": "rag",
        "retrieved_docs": retrieved_docs,
        "citations": dedupe_citations(citations),
        "context": context,
    }


def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve live web results and prepare grounded context for synthesis."""
    logger.info("Executing web_search_node")
    query = _query_from_state(state)

    if not web_search_tool.api_key:
        message = "TAVILY_API_KEY is not configured, so live web search is unavailable."
        logger.warning(message)
        return _error_message(message, "web_search")

    search_results = web_search_tool.search(query)
    if not search_results:
        message = "No live web results were returned for the query."
        return {
            "messages": [AIMessage(content=message)],
            "final_answer": message,
            "route": "web_search",
            "route_decision": "web_search",
            "search_results": [],
            "context": _append_context(state, mode="web_search", web_results=[], context_text=""),
        }

    formatted_context: List[str] = []
    citations: List[Dict[str, Any]] = []

    for index, result in enumerate(search_results, start=1):
        title = result.get("title", "No Title")
        url = result.get("url", "")
        snippet = result.get("content", "")
        formatted_context.append(f"[Web {index}] {title}\nURL: {url}\n{snippet}")
        citations.append(build_citation(url=url, source=title, page=None))

    context = _append_context(
        state,
        mode="web_search",
        web_results=search_results,
        context_text="\n\n".join(formatted_context),
    )

    return {
        "query": query,
        "route": "web_search",
        "route_decision": "web_search",
        "search_results": search_results,
        "citations": dedupe_citations(citations),
        "context": context,
    }


def url_reader_node(state: AgentState) -> Dict[str, Any]:
    """Read a URL from the query and prepare the page content for synthesis."""
    logger.info("Executing url_reader_node")
    query = _query_from_state(state)
    target_url = extract_url_from_text(query)

    if not target_url:
        message = "No valid URL was found in the query. Please include a full http:// or https:// link."
        return _error_message(message, "url_reader")

    try:
        url_data = url_reader_tool.read_url(target_url)
    except Exception as exc:
        logger.exception("URL reading failed: %s", exc)
        return _error_message(f"Failed to read the provided URL: {exc}", "url_reader")

    if url_data.get("method") == "error" or not url_data.get("content"):
        message = f"Failed to retrieve content from {target_url}: {url_data.get('content', 'No content returned.') }"
        return _error_message(message, "url_reader")

    citations = [build_citation(url=url_data.get("url", target_url), source=url_data.get("title", target_url), page=None)]
    context = _append_context(
        state,
        mode="url_reader",
        url_data=url_data,
        context_text=url_data.get("content", ""),
    )

    return {
        "query": query,
        "route": "url_reader",
        "route_decision": "url_reader",
        "url_content": url_data.get("content", ""),
        "citations": dedupe_citations(citations),
        "context": context,
    }


def general_llm_node(state: AgentState) -> Dict[str, Any]:
    """Handle general requests without retrieval."""
    logger.info("Executing general_llm_node")
    query = _query_from_state(state)

    try:
        llm = get_llm()
        messages = list(state.get("messages", []))
        if not messages or not isinstance(messages[-1], HumanMessage):
            messages.append(HumanMessage(content=query))

        response = llm.invoke(messages)
    except Exception as exc:
        logger.exception("General LLM call failed: %s", exc)
        return _error_message("The chat model is currently unavailable.", "general")

    context = _append_context(state, mode="general", answer=response.content)
    return {
        "messages": [response],
        "final_answer": response.content,
        "query": query,
        "route": "general",
        "route_decision": "general",
        "context": context,
    }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize a grounded answer from retrieved context and citations."""
    logger.info("Executing synthesis_node")
    query = _query_from_state(state)
    route = state.get("route") or state.get("route_decision") or "general"
    context = state.get("context", {})
    context_text = context.get("context_text", "")

    if not context_text:
        fallback_answer = state.get("final_answer") or context.get("error") or "No retrieval context was available for synthesis."
        return {
            "messages": [AIMessage(content=fallback_answer)],
            "final_answer": fallback_answer,
            "query": query,
            "route": route,
            "route_decision": route,
            "citations": dedupe_citations(state.get("citations", [])),
            "context": _append_context(state, answer=fallback_answer),
        }

    citations = dedupe_citations(state.get("citations", []))
    citation_lines = []
    for index, citation in enumerate(citations, start=1):
        citation_lines.append(
            f"[{index}] source={citation.get('source', '')}; url={citation.get('url', '')}; page={citation.get('page')}"
        )

    system_prompt = (
        "You are a senior research assistant. Use only the provided context to answer the query. "
        "If the context is insufficient, say so clearly. "
        "When relevant, refer to the numbered citations in your answer.\n\n"
        f"Route: {route}\n"
        f"Context:\n{context_text}\n\n"
        f"Citations:\n{chr(10).join(citation_lines) if citation_lines else 'None'}"
    )

    try:
        llm = get_llm()
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    except Exception as exc:
        logger.exception("Synthesis failed: %s", exc)
        return _error_message("I could not synthesize an answer from the retrieved context.", route)

    updated_context = _append_context(state, answer=response.content)
    return {
        "messages": [response],
        "final_answer": response.content,
        "query": query,
        "route": route,
        "route_decision": route,
        "citations": citations,
        "context": updated_context,
    }