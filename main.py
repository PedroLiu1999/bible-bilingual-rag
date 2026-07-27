import sys
from bible_rag.data.ingest import process_and_align
from bible_rag.vectorstore.indexer import run_indexing
from bible_rag.agent.rag_agent import BibleRAGAgent

def main():
    print("=== Bilingual Bible RAG Agent ===")
    if len(sys.argv) > 1 and sys.argv[1] == "--reindex":
        print("Re-running ingestion and vector database indexing...")
        process_and_align()
        run_indexing()

    agent = BibleRAGAgent()
    try:
        while True:
            query = input("\nEnter query (or \"exit\" to quit): ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                break
            if not query:
                continue
            print("\nThinking...")
            response = agent.run(query)
            print("\n" + "="*50)
            print(response)
            print("="*50)
    finally:
        agent.close()

if __name__ == "__main__":
    main()

