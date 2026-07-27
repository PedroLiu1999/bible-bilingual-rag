import json
import uuid

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bible_rag.config import COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL, PROCESSED_FILE

BATCH_SIZE = 512

def run_indexing():
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing {PROCESSED_FILE}. Run ingest first!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using execution device: {device.upper()}")

    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        verses = json.load(f)

    client = QdrantClient(path=str(DB_DIR))
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    for i in tqdm(range(0, len(verses), BATCH_SIZE), desc="Indexing Batches"):
        batch = verses[i : i + BATCH_SIZE]
        documents = [f"{v['book_name_en']} {v['chapter']}:{v['verse']} | {v['text_kjv']} | {v['text_cuv']}" for v in batch]
        embeddings = model.encode(documents, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

        points = []
        for idx, (verse_data, vector) in enumerate(zip(batch, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, verse_data["id"]))
            points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=verse_data))

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"\n Successfully indexed {len(verses)} verses into Qdrant using {device.upper()}!")

if __name__ == "__main__":
    run_indexing()
