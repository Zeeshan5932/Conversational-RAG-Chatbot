```markdown
# 🤖 RAGentic AI

> An Enterprise-Grade Multi-Agent Hybrid RAG System powered by **LangGraph**, **Groq**, **Gemini Embeddings**, **ChromaDB**, and **FastAPI**.

---

## 📋 Overview

**RAGentic AI** is a production-ready, stateful Retrieval-Augmented Generation (RAG) assistant designed for deep multi-document intelligence and accurate contextual Q&A. Powered by **LangGraph** orchestration, the system dynamically routes queries between multi-document retrieval nodes and general LLM reasoning agents. 

It leverages **Groq (`llama-3.3-70b-versatile`)** for high-speed inference, **Google Gemini (`text-embedding-004`)** for precise semantic vectorization, and **ChromaDB** for persistent vector storage.

---

## ✨ Key Features

- **🔀 Intelligent Agentic Routing:** Automatically evaluates incoming user queries to route between local document knowledge bases (RAG) and general reasoning agents.
- **📚 Multi-Document PDF Ingestion:** Upload and parse multiple complex PDF documents with source tracking and inline citations.
- **⚡ Blazing Fast Inference:** Powered by Groq LPU hardware acceleration running Llama 3.3 70B.
- **💾 Persistent Thread Memory:** Uses LangGraph `MemorySaver` checkpointer for stateful, multi-turn chat conversations via `thread_id`.
- **🎯 Robust Embedding Normalization:** Handles Google Generative AI model string normalization gracefully to prevent missing model runtime errors.
- **🎨 Interactive Streamlit UI:** Sleek, user-friendly frontend with document context viewers and real-time citation rendering.

---

## 🏗️ Tech Stack & Architecture

- **Orchestration:** LangGraph / LangChain
- **LLM Engine:** Groq (`llama-3.3-70b-versatile`)
- **Embedding Model:** Google Gemini (`text-embedding-004`)
- **Vector Store:** ChromaDB (Local Persistent Storage)
- **Backend API:** FastAPI / Uvicorn
- **Frontend UI:** Streamlit
- **PDF Loader:** PyPDFLoader

---

## 🛠️ Project Structure

```text
├── app/
│   ├── api/             # FastAPI endpoints and route handlers
│   ├── agent/           # LangGraph state, nodes, router, and workflow graph
│   │   ├── graph.py     # Core agent state graph definition
│   │   ├── nodes.py     # Execution nodes (RAG, General LLM, Synthesis)
│   │   ├── router.py    # Intent classification logic
│   │   └── state.py     # Custom AgentState schema
│   ├── rag/             # RAG engine implementation
│   │   ├── embeddings.py# Normalized Google Gemini embedding factory
│   │   └── vectorstore.py# ChromaDB ingestion & retrieval utilities
│   └── config.py        # Environment variables & system configuration
├── data/                # Persistent ChromaDB storage directory
├── frontend/            # Streamlit dashboard interface
├── .env                 # API Keys & model environment settings
├── main.py              # FastAPI server entry point
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/ragentic-ai.git](https://github.com/your-username/ragentic-ai.git)
cd ragentic-ai

```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Models Configuration
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=text-embedding-004

# Vector Store Config
CHROMA_PERSIST_DIR=./data/chroma_db

```

---

## 🚀 Running the Application

### Start the Backend (FastAPI)

```bash
uvicorn app.main:app --reload
# Server will run on http://localhost:8000

```

### Start the Frontend (Streamlit)

```bash
streamlit run frontend/app.py
# UI will run on http://localhost:8501

```

---

## 🧪 Quick Test Workflow

1. Open the **Streamlit UI** at `http://localhost:8501`.
2. Upload one or multiple PDF files using the sidebar uploader.
3. Ask document-specific questions (e.g., *"What are the core responsibilities outlined in the project structure?"*).
4. Observe the system retrieve context from **ChromaDB**, synthesize the answer using **Llama 3.3**, and provide inline source citations!

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

```

```