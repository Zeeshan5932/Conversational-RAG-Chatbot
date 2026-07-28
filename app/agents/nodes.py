from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.agents.state import AgentState
from app.agents.router import QueryRouter
from app.llm.gemini import get_gemini_llm
from app.rag.vectorstore import VectorStoreManager
from app.rag.retriever import DocumentRetriever
from app.tools.web_search import WebSearchTool
from app.tools.url_reader import URLReaderTool, extract_url_from_text
from app.utils.logger import logger

router_instance = QueryRouter()
llm = get_gemini_llm(temperature=0.3)
web_search_tool = WebSearchTool(max_results=5)
url_reader_tool = URLReaderTool()


def router_node(state: AgentState) -> Dict[str, Any]:
    """Graph node that evaluates the user query and assigns the route decision."""
    query = state["user_query"]
    decision = router_instance.route_query(query)
    return {"route_decision": decision.route}


def general_llm_node(state: AgentState) -> Dict[str, Any]:
    """Handles general questions directly via Gemini without tool search."""
    logger.info("Executing general_llm_node...")
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(content=state["user_query"]))

    response = llm.invoke(messages)
    return {
        "messages": [response],
        "final_answer": response.content
    }


def rag_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves relevant chunks from ChromaDB and synthesizes an answer."""
    logger.info("Executing rag_node...")
    query = state["user_query"]
    
    vector_store_mgr = VectorStoreManager()
    retriever = DocumentRetriever(vector_store_mgr, k=4)
    docs = retriever.retrieve(query)
    
    formatted_context = ""
    retrieved_docs_metadata = []
    
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", 1)
        formatted_context += f"\n--- Document [{idx}] ({source}, Page {page}) ---\n{doc.page_content}\n"
        retrieved_docs_metadata.append({
            "source": source,
            "page": page,
            "content": doc.page_content[:200]
        })

    system_prompt = (
        "You are an accurate research assistant. Answer the user question based strictly on "
        "the provided document context below. If the context does not contain enough information, "
        "clearly state that context is insufficient.\n\n"
        f"Context:\n{formatted_context}"
    )
    
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(prompt_messages)
    
    return {
        "messages": [response],
        "final_answer": response.content,
        "retrieved_docs": retrieved_docs_metadata
    }


def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Queries live web data via Tavily and synthesizes a grounded answer with citations."""
    logger.info("Executing web_search_node...")
    query = state["user_query"]
    
    search_results = web_search_tool.search(query)
    
    if not search_results:
        msg = "Unable to retrieve real-time web results (API key missing or search returned no results)."
        return {
            "final_answer": msg,
            "messages": [AIMessage(content=msg)],
            "search_results": []
        }

    formatted_search_context = ""
    citations = []
    
    for idx, res in enumerate(search_results, start=1):
        title = res.get("title", "No Title")
        url = res.get("url", "")
        snippet = res.get("content", "")
        formatted_search_context += f"\n--- Web Source [{idx}]: {title} ({url}) ---\n{snippet}\n"
        citations.append({"title": title, "url": url})

    system_prompt = (
        "You are an AI Web Research Assistant. Synthesize a comprehensive and factual response "
        "to the user query using the live web search results below. Always attribute key claims "
        "using inline numbers corresponding to the web sources (e.g., [1], [2]).\n\n"
        f"Live Web Search Results:\n{formatted_search_context}"
    )

    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = llm.invoke(prompt_messages)

    return {
        "messages": [response],
        "final_answer": response.content,
        "search_results": search_results,
        "citations": citations
    }


def url_research_node(state: AgentState) -> Dict[str, Any]:
    """Extracts URL content from target page and summarizes or answers user query."""
    logger.info("Executing url_research_node...")
    query = state["user_query"]
    target_url = extract_url_from_text(query)

    if not target_url:
        msg = "No valid URL found in your query. Please provide a full URL (e.g., https://example.com)."
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

    url_data = url_reader_tool.read_url(target_url)
    
    if "Error" in url_data["title"] or not url_data["content"]:
        msg = f"Failed to retrieve content from {target_url}: {url_data['content']}"
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

    system_prompt = (
        "You are a webpage content analyst. Analyze the provided webpage content and "
        "answer the user query accurately. Always cite the URL as the primary source.\n\n"
        f"URL: {url_data['url']}\n"
        f"Page Title: {url_data['title']}\n"
        f"Webpage Content:\n{url_data['content']}"
    )

    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = llm.invoke(prompt_messages)

    return {
        "messages": [response],
        "final_answer": response.content,
        "url_content": url_data["content"],
        "citations": [{"title": url_data["title"], "url": url_data["url"]}]
    }


def hybrid_node(state: AgentState) -> Dict[str, Any]:
    """Executes local vector store search and web search in parallel to form combined context."""
    logger.info("Executing hybrid_node (Document RAG + Live Web Search)...")
    query = state["user_query"]

    # 1. Internal Vector RAG Retrieval
    vector_store_mgr = VectorStoreManager()
    retriever = DocumentRetriever(vector_store_mgr, k=3)
    docs = retriever.retrieve(query)

    formatted_doc_context = ""
    retrieved_docs_metadata = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", 1)
        formatted_doc_context += f"\n--- Internal Document [{idx}] ({source}, Page {page}) ---\n{doc.page_content}\n"
        retrieved_docs_metadata.append({
            "source": source,
            "page": page,
            "content": doc.page_content[:200]
        })

    # 2. Live Web Search Retrieval
    search_results = web_search_tool.search(query)
    formatted_web_context = ""
    citations = []
    for idx, res in enumerate(search_results, start=1):
        title = res.get("title", "No Title")
        url = res.get("url", "")
        snippet = res.get("content", "")
        formatted_web_context += f"\n--- External Web Source [{idx}]: {title} ({url}) ---\n{snippet}\n"
        citations.append({"title": title, "url": url})

    # 3. Combine Contexts and Prompt Gemini
    system_prompt = (
        "You are an expert hybrid research system. You have been provided with both internal document "
        "knowledge base excerpts and real-time external web search results.\n\n"
        "Synthesize a clear, detailed, and comprehensive response. Clearly distinguish between "
        "insights gathered from internal documents versus live web sources.\n\n"
        f"=== INTERNAL DOCUMENTS ===\n{formatted_doc_context if formatted_doc_context else 'No internal documents found.'}\n\n"
        f"=== EXTERNAL LIVE WEB RESULTS ===\n{formatted_web_context if formatted_web_context else 'No external web results found.'}"
    )

    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = llm.invoke(prompt_messages)

    return {
        "messages": [response],
        "final_answer": response.content,
        "retrieved_docs": retrieved_docs_metadata,
        "search_results": search_results,
        "citations": citations
    }