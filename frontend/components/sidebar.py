import streamlit as st
import requests


def render_sidebar(backend_url: str):
    """Renders the Streamlit sidebar including system status and document uploader."""
    with st.sidebar:
        st.header("⚙️ Configuration & Upload")
        
        # System Health Status
        try:
            response = requests.get(f"{backend_url}/api/v1/health", timeout=3)
            if response.status_code == 200:
                st.success("Backend API Connected")
            else:
                st.error("Backend Unhealthy")
        except Exception:
            st.error("Backend Unreachable")

        st.divider()

        # File Upload Section
        st.subheader("📄 Upload Knowledge Base Document")
        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, TXT, PPT, or PPTX",
            type=["pdf", "docx", "txt","ppt", "pptx"],
            help="Uploaded documents will be chunked and indexed into ChromaDB for RAG retrieval."
        )

        if uploaded_file is not None:
            if st.button("Process & Index Document", use_container_width=True):
                with st.spinner("Ingesting document into vector store..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        res = requests.post(f"{backend_url}/api/v1/documents/upload", files=files, timeout=60)
                        
                        if res.status_code == 201:
                            data = res.json()
                            st.success(f"Indexed **{data['filename']}**!")
                            st.caption(f"Pages: {data['pages']} | Chunks Indexed: {data['chunks']}")
                        else:
                            st.error(f"Error: {res.json().get('detail', 'Upload failed')}")
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")