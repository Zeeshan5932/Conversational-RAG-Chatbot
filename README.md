Here is the complete, comprehensive `README.md` file tailored specifically to your project's architecture, including all 10 implementation phases, full feature breakdowns, environment setup, and instructions for running FastAPI and Streamlit both separately and together.

---

### `README.md`

```markdown
# 🤖 Multi-Agent RAG & Live Web Research Assistant

An enterprise-grade, stateful multi-agent research system built with **LangGraph**, **Google Gemini 2.5**, **ChromaDB**, **FastAPI**, **Tavily**, **Firecrawl**, and **Streamlit**.

The system dynamically classifies user intent into distinct specialized processing nodes—from local vector retrieval and real-time live web search to deep URL content analysis and hybrid synthesis—while maintaining persistent conversation memory using stateful checkpointers.

---

## 🌟 Key Features

- **Intelligent Query Routing**: Classifies queries using structured Pydantic outputs into 5 execution paths:
  1. `general_llm`: Direct reasoning via Gemini 2.5 without tool retrieval.
  2. `rag`: Local document retrieval over vector embeddings stored in ChromaDB.
  3. `web_search`: Live internet search and citations powered by Tavily API.
  4. `url_research`: Direct web page scraping and content extraction (Firecrawl with BeautifulSoup fallback).
  5. `hybrid`: Simultaneous execution of internal document retrieval and external live web search.
- **Stateful Thread Memory**: Multi-turn conversation persistence across chat sessions powered by LangGraph `MemorySaver`.
- **Citations & Attribution**: Structured source tracking returning document metadata (source file, page numbers) and web URLs.
- **Modular Frontend**: Decoupled Streamlit frontend architecture (`frontend/streamlit_app.py` and UI components).
- **RESTful API**: Production-ready FastAPI backend with OpenAPI documentation and CORS support.
- **Containerized Deployment**: Ready-to-use `Dockerfile` and `docker-compose.yml`.

---

## 📂 Project Structure

```text
├── app/
│   ├── agents/
│   │   ├── graph.py         # StateGraph build & MemorySaver checkpointer
│   │   ├── nodes.py         # Agent execution nodes (RAG, Web, URL, Hybrid, General)
│   │   ├── router.py        # Query routing logic using Gemini structured outputs
│   │   └── state.py         # AgentState TypedDict schema
│   ├── api/
│   │   └── routes.py        # FastAPI API endpoints (/chat, /documents/upload, /health)
│   ├── config.py            # Pydantic settings & environment configuration
│   ├── llm/
│   │   └── gemini.py        # Gemini 2.5 LLM initializer
│   ├── rag/
│   │   ├── loader.py        # Document loading (PDF, DOCX, TXT)
│   │   ├── retriever.py     # ChromaDB similarity retriever
│   │   └── vectorstore.py   # VectorStore lifecycle management
│   ├── services/
│   │   ├── chat_service.py  # LangGraph graph execution wrapper
│   │   └── document_service.py # Document ingestion pipeline service
│   ├── tools/
│   │   ├── url_reader.py    # URL scraping (Firecrawl / BeautifulSoup fallback)
│   │   └── web_search.py    # Tavily web search wrapper
│   └── main.py              # FastAPI application initialization
├── frontend/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat.py          # Chat message history & user input rendering
│   │   ├── sidebar.py       # Knowledge base upload & operations panel
│   │   └── sources.py       # Expandable citation & reference block
│   └── streamlit_app.py     # Streamlit entry point
├── tests/
│   └── test_agent.py        # Unit and integration test suite
├── .env.example             # Template environment variables
├── Dockerfile               # Production container definition
├── docker-compose.yml       # Multi-container service definition
├── requirements.txt         # Python dependencies
└── run.py                   # Unified launcher script (FastAPI + Streamlit)

```

---

## 🛠️ Prerequisites & Setup

### 1. Clone Repository & Create Virtual Environment

```bash
git clone [https://github.com/your-username/multi-agent-rag.git](https://github.com/your-username/multi-agent-rag.git)
cd multi-agent-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
# Core API Keys
GEMINI_API_KEY=your_actual_gemini_api_key
TAVILY_API_KEY=your_actual_tavily_api_key

# Optional API Keys
FIRECRAWL_API_KEY=your_optional_firecrawl_api_key

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
GEMINI_MODEL=gemini-2.5-flash
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads

```

---

## 🚀 How to Run the Application

You can run the backend and frontend together, separately, or via Docker.

### Option A: Run Both Together (Unified Launcher)

The `run.py` script starts the FastAPI backend and Streamlit UI in parallel:

```bash
python run.py

```

* **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
* **FastAPI Backend**: [http://localhost:8000](http://localhost:8000)
* **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Run Services Separately

If you prefer to run and debug the backend and frontend in separate terminal windows:

#### Terminal 1: Run FastAPI Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

#### Terminal 2: Run Streamlit Frontend

```bash
streamlit run frontend/streamlit_app.py --server.port 8501

```

---

### Option C: Run via Docker Compose

To build and launch the entire application stack in isolated containers:

```bash
# Build and start services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down

```

---

## 📡 API Reference & Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/v1/health` | `GET` | Health check verifying model configuration and system status. |
| `/api/v1/documents/upload` | `POST` | Ingests PDF, DOCX, or TXT files into the ChromaDB vector database. |
| `/api/v1/chat` | `POST` | Main endpoint executing user queries through the stateful LangGraph engine. |
| `/api/v1/research/url` | `POST` | Direct webpage content extraction and preview endpoint. |

### Sample Chat Payload (`POST /api/v1/chat`)

```json
{
  "message": "What are the latest developments in AI in 2026?",
  "thread_id": "session_abc123"
}

```

---

## 🧪 Testing

Run unit and integration tests using `pytest`:

```bash
pytest tests/ -v

`

```