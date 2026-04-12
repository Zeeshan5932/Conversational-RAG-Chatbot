import os
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Conversational RAG Chatbot", page_icon=":satellite:", layout="wide")

st.markdown(
    """
<style>
    :root {
        --bg-1: #f2efe8;
        --bg-2: #dce9df;
        --ink: #132014;
        --accent: #0d6e4f;
        --card: rgba(255, 255, 255, 0.72);
    }
    .stApp {
        background: radial-gradient(circle at 10% 10%, var(--bg-2), var(--bg-1) 60%);
    }
    .hero {
        border: 1px solid rgba(19, 32, 20, 0.1);
        background: var(--card);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(6px);
    }
    .hero h1 {
        color: var(--ink);
        margin: 0;
        font-size: 1.9rem;
    }
    .hero p {
        margin: 0.2rem 0 0;
        color: #2b3b2d;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Conversational RAG Chatbot</h1>
  <p>Ask any question. App fetches live web context, then responds with citations.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Realtime Settings")
    model_name = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], index=0)
    temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    max_sources = st.slider("Max web sources", min_value=3, max_value=10, value=6, step=1)
    st.caption("Tip: lower creativity gives more factual style answers.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I can answer general questions and fetch latest info from the web.",
            "sources": [],
        }
    ]


def fetch_live_results(query: str, limit: int) -> list[dict]:
    results = []
    seen_links = set()
    search_queries = [query, f"{query} latest update"]

    for q in search_queries:
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(q, max_results=limit):
                    link = item.get("href", "").strip()
                    if not link or link in seen_links:
                        continue
                    seen_links.add(link)
                    results.append(
                        {
                            "title": item.get("title", "Untitled"),
                            "snippet": item.get("body", ""),
                            "url": link,
                        }
                    )
                    if len(results) >= limit:
                        return results
        except Exception:
            continue

    return results


def build_context(sources: list[dict]) -> str:
    if not sources:
        return "No live web sources were found."

    chunks = []
    for idx, src in enumerate(sources, start=1):
        chunks.append(
            f"[{idx}] Title: {src['title']}\n"
            f"URL: {src['url']}\n"
            f"Snippet: {src['snippet']}"
        )
    return "\n\n".join(chunks)


def answer_query(query: str, sources: list[dict]) -> str:
    client = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=temperature,
    )

    today = datetime.now().strftime("%Y-%m-%d")
    context_text = build_context(sources)

    prompt = (
        "You are a factual assistant with web context.\n"
        f"Today's date: {today}\n"
        "Rules:\n"
        "1) Use provided web sources for real-time or factual claims.\n"
        "2) If data is uncertain or sources are weak, clearly mention uncertainty.\n"
        "3) For prices/rates, include currency/unit and mention that values can change quickly.\n"
        "4) Cite source numbers like [1], [2] where relevant.\n\n"
        f"User query: {query}\n\n"
        f"Web context:\n{context_text}\n\n"
        "Provide a clear answer first, then a short 'Sources used' line with citation numbers."
    )

    response = client.invoke(prompt)
    return response.content


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View web sources"):
                for idx, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"{idx}. [{src['title']}]({src['url']})")
                    if src["snippet"]:
                        st.caption(src["snippet"])


query = st.chat_input("Ask anything: petrol price, sports score, news, weather, coding, facts...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query, "sources": []})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching live web and generating answer..."):
            sources = fetch_live_results(query, max_sources)
            final_answer = answer_query(query, sources)

        st.markdown(final_answer)
        if sources:
            with st.expander("View web sources"):
                for idx, src in enumerate(sources, start=1):
                    st.markdown(f"{idx}. [{src['title']}]({src['url']})")
                    if src["snippet"]:
                        st.caption(src["snippet"])

    st.session_state["messages"].append(
        {"role": "assistant", "content": final_answer, "sources": sources}
    )