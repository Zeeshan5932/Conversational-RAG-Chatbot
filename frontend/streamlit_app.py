import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import streamlit as st
import os
from frontend.components.sidebar import render_sidebar
from frontend.components.chat import render_chat_interface

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAGentic AI")
st.caption("Conversational RAG & Web Search Engine")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Render Sidebar Component
render_sidebar(BACKEND_URL)
# Render Interactive Chat Interface
render_chat_interface(BACKEND_URL)
st.divider()
st.info("Phase 2 complete: Document parsing, chunking, Gemini Embeddings, and ChromaDB indexing active.")