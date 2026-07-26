import streamlit as st
import requests
import os

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Assistant")
st.caption("Conversational RAG & Web Search Engine")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Sidebar Configuration
with st.sidebar:
    st.header("System Status")
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            st.success("Backend Connected")
            st.json(data)
        else:
            st.error("Backend unreachable")
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")

st.divider()
st.info("Phase 1 initialization complete. Backend API and baseline UI active.")