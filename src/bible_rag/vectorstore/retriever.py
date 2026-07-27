import re
from typing import Any

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

from bible_rag.config import COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL, QDRANT_URL, QDRANT_API_KEY

BOOK_MAP = {
    "gen": "Genesis", "創": "Genesis", "創世記": "Genesis",
    "exo": "Exodus", "出": "Exodus", "出埃及記": "Exodus",
    "mat": "Matthew", "太": "Matthew", "馬太福音": "Matthew",
    "jhn": "John", "john": "John", "約": "John", "約翰福音": "John",
    "rom": "Romans", "羅": "Romans", "羅馬書": "Romans",
    "rev": "Revelation", "啟": "Revelation", "啟示錄": "Revelation",
    "1jn": "1 John", "1john": "1 John", "約一": "1 John", "約翰一書": "1 John",
}

class BibleRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if QDRANT_URL:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self.client = QdrantClient(path=str(DB_DIR))
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

    def close(self):
        if hasattr(self, "client") and self.client is not None:
            try: self.client.close()
            except Exception: pass

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

    def parse_verse_reference(self, query: str) -> dict[str, Any] | None:
        pattern = r"([1-3]?\s*[\u4e00-\u9fa5a-zA-Z]+)\s*(\d+)[:：](\d+)"
        match = re.search(pattern, query)
        if not match: return None

        book_raw, chapter, verse = match.groups()
        book_clean = book_raw.strip().lower().replace(" ", "")
        canonical_book = BOOK_MAP.get(book_clean, book_raw.strip().title())

        return {"book": canonical_book, "chapter": int(chapter), "verse": int(verse)}

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        ref = self.parse_verse_reference(query)
        if ref:
            scroll_res = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="book_name_en", match=MatchValue(value=ref["book"])),
                        FieldCondition(key="chapter", match=MatchValue(value=ref["chapter"])),
                        FieldCondition(key="verse", match=MatchValue(value=ref["verse"])),
                    ]
                ),
                limit=1,
            )[0]
            if scroll_res:
                payload = scroll_res[0].payload
                payload["retrieval_type"] = "exact_match"
                return [payload]

        query_vector = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()
        results = self.client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k).points

        retrieved = []
        for res in results:
            item = res.payload
            item["score"] = res.score
            item["retrieval_type"] = "semantic"
            retrieved.append(item)
        return retrieved
