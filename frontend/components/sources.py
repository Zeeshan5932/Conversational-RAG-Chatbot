import streamlit as st
from typing import List, Dict, Any

def render_sources(citations: List[Dict[str, Any]]):
    """Renders formatted citations and document source metadata in an expandable UI block."""
    if not citations:
        return

    with st.expander("🔗 Source Citations & References", expanded=False):
        for idx, cite in enumerate(citations, start=1):
            if "url" in cite and cite["url"]:
                title = cite.get("title", cite["url"])
                st.markdown(f"**[{idx}]** [{title}]({cite['url']})")
            elif "source" in cite and cite["source"]:
                source_file = cite.get("source", "Unknown Document")
                page = cite.get("page", 1)
                st.markdown(f"**[{idx}]** Document: `{source_file}` *(Page {page})*")
            else:
                st.markdown(f"**[{idx}]** Unknown source metadata.")