import os
import json
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from ddgs import DDGS
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

st.set_page_config(
    page_title="Conversational RAG Chatbot",
    page_icon=":satellite:",
    layout="wide"
)

st.markdown(
    """
<style>
    :root {
        --bg-1: #0A0F2C;
        --bg-2: #121a3f;
        --glass: rgba(255, 255, 255, 0.06);
        --border: rgba(255, 255, 255, 0.08);
        --text: #E6EAF2;
        --accent: #3B82F6;
    }

    .stApp {
        background: radial-gradient(circle at 20% 20%, #121a3f, #050816 80%);
    }

    .hero {
        background: var(--glass);
        border: 1px solid var(--border);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }

    .hero p {
        color: #b8c1ec;
        margin-top: 4px;
    }

    [data-testid="stChatMessage"] {
        background: var(--glass);
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 10px;
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: #050816;
        border-right: 1px solid var(--border);
    }

    .block-container {
        padding-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Conversational RAG Chatbot</h1>
  <p>Ask any question. The assistant can search the web in real time when needed.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Realtime Settings")
    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
        index=0
    )
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
    max_sources = st.slider("Max web sources", 3, 10, 6, 1)
    st.caption("Lower creativity usually gives more factual answers.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything — I can search live web data when needed."}
    ]


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information and return concise results with title, snippet, and URL."""
    results = []
    seen = set()

    search_queries = [query, f"{query} latest update"]

    for q in search_queries:
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(q, max_results=max_sources):
                    url = (item.get("href") or "").strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)

                    results.append({
                        "title": item.get("title", "Untitled"),
                        "snippet": item.get("body", ""),
                        "url": url
                    })

                    if len(results) >= max_sources:
                        return json.dumps(results, ensure_ascii=False)
        except Exception:
            continue

    return json.dumps(results, ensure_ascii=False)


def run_agent(user_query: str):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=temperature,
    )

    llm_with_tools = llm.bind_tools([web_search])

    today = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
You are a helpful factual assistant.

Today's date: {today}

Rules:
1. Use the web_search tool whenever the user asks for recent, live, changing, or factual information.
2. If the answer is stable general knowledge, you may answer directly.
3. When tool results are used, cite them inline like [1], [2].
4. After the answer, add a short section called 'Sources'.
5. If the web results are weak or uncertain, say so clearly.
"""

    messages = [
        HumanMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    first_response = llm_with_tools.invoke(messages)

    collected_sources = []

    if getattr(first_response, "tool_calls", None):
        tool_messages = []

        for tool_call in first_response.tool_calls:
            if tool_call["name"] == "web_search":
                tool_result = web_search.invoke(tool_call["args"])
                parsed = json.loads(tool_result) if tool_result else []
                collected_sources.extend(parsed)

                tool_messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

        final_response = llm_with_tools.invoke(messages + [first_response] + tool_messages)
        return final_response.content, collected_sources

    return first_response.content, collected_sources


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View web sources"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"{i}. [{src['title']}]({src['url']})")
                    if src.get("snippet"):
                        st.caption(src["snippet"])


query = st.chat_input("Ask anything: news, sports, weather, prices, coding, facts...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching the web if needed..."):
            answer, sources = run_agent(query)

        st.markdown(answer)

        if sources:
            with st.expander("View web sources"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"{i}. [{src['title']}]({src['url']})")
                    if src.get("snippet"):
                        st.caption(src["snippet"])

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )