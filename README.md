# 📖 Bilingual Bible RAG Agent (CUV & KJV)

A production-grade Retrieval-Augmented Generation (RAG) agent for exploring and querying the Bible simultaneously in **Chinese (Chinese Union Version - CUV / 和合本)** and **English (King James Version - KJV)**.

Built to demonstrate senior AI engineering patterns:
- **Verse-Atomic Cross-Lingual Alignment:** Prevents mid-sentence verse splitting by indexing complete, aligned CUV/KJV records.
- **Hybrid Intent Routing:** Uses regex for deterministic exact verse lookups (0ms, 100% precision) with automatic fallback to dense vector search for conceptual/topical queries.
- **GPU-Accelerated Vector Indexing:** PyTorch CUDA-backed sentence embeddings (`paraphrase-multilingual-mpnet-base-v2`) inside a local **Qdrant** vector store.
- **Stateful Agentic Workflow:** LangGraph execution pipeline with strict grounded system prompts to prevent hallucination.
- **Open-Weights NIM LLM Execution:** Powered by NVIDIA Inference Microservices (`meta/llama-3.3-70b-instruct` / `deepseek-r1`).
- **Full-Stack Application:** Includes an interactive web interface powered by FastAPI, Tailwind CSS, and Alpine.js.

---

## 🛠️ Tech Stack & Dependencies

- **Orchestration:** LangGraph / LangChain
- **LLM Provider:** NVIDIA NIM (`langchain-nvidia-ai-endpoints`)
- **Vector Database:** [Qdrant](https://qdrant.tech/) (Local file-system persistent mode)
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (PyTorch CUDA 12.6)
- **Web Backend & UI:** FastAPI, Uvicorn, Tailwind CSS, Alpine.js, Marked.js
- **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (Fast, modern Python environment management)

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- `uv` installed (`pip install uv` or `winget install astral-sh.uv`)
- NVIDIA GPU with CUDA drivers (optional, automatically falls back to CPU)

### 1. Installation & Environment Setup

Clone the repository and synchronize dependencies using `uv`:

```powershell
# Sync project dependencies
uv sync

# Install the local src package in editable mode
uv pip install -e .

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your NVIDIA API key (or OpenAI key if using ChatOpenAI):

```env
NVIDIA_API_KEY=nvapi-your_actual_key_here
MODEL_NAME=meta/llama-3.3-70b-instruct

```

> **Note:** Free API keys and complimentary credits are available at [build.nvidia.com](https://build.nvidia.com/).

---

## 📥 Pipeline & Application Execution

### Step 1: Initial Data Ingestion & GPU Vector Indexing

Run the initial ingestion script to download raw CUV/KJV JSON files, align them by verse, and create the vector database index:

```powershell
uv run main.py --reindex

```

*(Performance Note: GPU execution embeds the entire Bible—~31,102 verses—in ~15 seconds.)*

---

### Step 2: Launch the Web Application (FastAPI + Tailwind UI)

Start the production FastAPI web server:

```powershell
uv run python app.py

```

Open your web browser and navigate to:
👉 **`http://127.0.0.1:8000`**

#### Key Web Features:

* **Interactive Chat Interface:** Formatted side-by-side CUV/KJV verse comparison cards.
* **REST API Endpoint:** Send POST requests to `/api/chat` with `{ "query": "..." }`.
* **Markdown & Code Rendering:** Clean display of bullet points, bolding, and scripture citations.

---

### Step 3: Terminal CLI Mode (Alternative)

If you prefer to test queries directly in your terminal:

```powershell
uv run main.py

```

---

## 📁 Deep Modular Directory Structure

```text
bible-bilingual-rag/
├── .env                     # Secrets and configurations (NVIDIA_API_KEY)
├── .gitignore               # Protects .env, qdrant_db, and data directories
├── app.py                   # FastAPI application & embedded web frontend
├── main.py                  # Entry point CLI runner
├── pyproject.toml           # Package configuration & CUDA wheel indexes
├── README.md                # Project documentation
├── data/                    
│   ├── raw/                 # Cached raw KJV and CUV JSON downloads
│   └── processed/           # Aligned verse-by-verse dataset
├── qdrant_db/               # Persistent local Qdrant vector database
└── src/
    └── bible_rag/           # Core application package
        ├── config.py        # Centralized pathing, models, and .env loading
        ├── data/
        │   └── ingest.py    # Data downloader & cross-lingual aligner
        ├── vectorstore/
        │   ├── indexer.py   # GPU-accelerated Qdrant upsert pipeline
        │   └── retriever.py # Hybrid router (Regex Entity + Vector Search)
        └── agent/
            └── rag_agent.py # LangGraph agentic state graph & system prompts

```

---

## 🧪 Example Test Queries

Try these queries in either the web interface or terminal CLI to test the pipeline:

| Intent Type | Example Query | Expected Behavior |
| --- | --- | --- |
| **Exact Reference** | `"What does John 3:16 say?"` | Triggers **Regex Router** -> Direct payload fetch (0ms). |
| **Chinese Reference** | `"約翰福音 3:16 說了什麼？"` | Triggers **Regex Router** -> Direct payload fetch (0ms). |
| **Conceptual Search** | `"What are the biblical teachings on peacemakers?"` | Triggers **Vector Search** -> Retrieves Matthew 5:9. |
| **Cross-Lingual** | `"關於饒恕和寬恕的教導"` | Triggers **Vector Search** -> Retrieves Mark 11:26, 2 Cor 2:10. |

```
