import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

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

_raw_key = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_KEY = _raw_key.strip().strip("\"'").strip()

DEFAULT_LLM_MODEL = os.getenv("MODEL_NAME", "meta/llama-3.3-70b-instruct")
