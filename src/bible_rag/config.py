import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(find_dotenv(), override=True)

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = BASE_DIR / "qdrant_db"

PROCESSED_FILE = PROCESSED_DIR / "aligned_cuv_kjv.json"
COLLECTION_NAME = "bible_bilingual"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").lower().strip()

# API Keys & Hosts
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip().strip("\"'").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip("\"'").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

# Model defaults per provider
DEFAULT_LLM_MODEL = os.getenv("MODEL_NAME")
if not DEFAULT_LLM_MODEL:
    if LLM_PROVIDER == "ollama":
        DEFAULT_LLM_MODEL = "llama3.2"
    elif LLM_PROVIDER == "openai":
        DEFAULT_LLM_MODEL = "gpt-4o-mini"
    else:
        DEFAULT_LLM_MODEL = "meta/llama-3.3-70b-instruct"
