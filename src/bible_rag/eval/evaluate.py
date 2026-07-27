# src/bible_rag/eval/evaluate.py
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from bible_rag.agent.rag_agent import BibleRAGAgent
from bible_rag.config import NVIDIA_API_KEY, DEFAULT_LLM_MODEL, BASE_DIR

# Golden Dataset for Bilingual Theological Benchmarking
GOLDEN_DATASET = [
    {
        "question": "What does John 3:16 say?",
        "ground_truth": "John 3:16 states that God so loved the world that He gave His only begotten Son, that whoever believes in Him should not perish but have everlasting life. (約翰福音 3:16: 神愛世人，甚至將祂的獨生子賜給他們...)",
    },
    {
        "question": "What are the blessings for peacemakers in the Sermon on the Mount?",
        "ground_truth": "According to Matthew 5:9, peacemakers are blessed because they shall be called the children of God. (使人和睦的人有福了！因為他們必稱為神的兒女。)",
    },
    {
        "question": "What does the Bible teach about forgiveness in Mark 11:25-26?",
        "ground_truth": "Mark 11:25-26 teaches that when you stand praying, you must forgive others so that your Father in heaven may also forgive your trespasses.",
    },
    {
        "question": "關於饒恕和寬恕，聖經有什麼教導？",
        "ground_truth": "聖經教導我們要彼此赦免，如哥林多後書 2:10 與馬可福音 11:25-26 所言，若不饒恕人，天父也不饒恕我們的過犯。",
    },
]


def run_ragas_evaluation():
    print("🚀 Initializing Ragas Evaluation Harness...")

    if not NVIDIA_API_KEY:
        raise ValueError("❌ Missing NVIDIA_API_KEY in environment!")

    # 1. Instantiate Evaluation Judge Models via NVIDIA NIM
    eval_llm = ChatNVIDIA(
        model=DEFAULT_LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=0.0
    )
    eval_embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=NVIDIA_API_KEY
    )

    # 2. Run your live Agent on the Golden Benchmark
    agent = BibleRAGAgent()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\n⏳ Executing RAG Agent against Golden Dataset...")
    try:
        for item in GOLDEN_DATASET:
            q = item["question"]
            gt = item["ground_truth"]

            # Retrieve context and generate response
            verses = agent.retriever.retrieve(q, top_k=3)
            formatted_contexts = [
                f"{v['book_name_en']} {v['chapter']}:{v['verse']} - KJV: {v['text_kjv']} | CUV: {v['text_cuv']}"
                for v in verses
            ]
            response = agent.run(q)

            questions.append(q)
            answers.append(response)
            contexts.append(formatted_contexts)
            ground_truths.append(gt)
    finally:
        agent.close()

    # 3. Construct HuggingFace Dataset required by Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data_dict)

    # 4. Configure metrics with explicit evaluator LLMs
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    for metric in metrics:
        metric.llm = eval_llm
        if hasattr(metric, "embeddings") and metric.embeddings is None:
            metric.embeddings = eval_embeddings

    # 5. Execute Ragas Evaluation
    print("\n📊 Computing Ragas Metrics (Faithfulness, Relevancy, Precision, Recall)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # 6. Save & Display Results
    df = result.to_pandas()
    eval_output_dir = BASE_DIR / "eval_results"
    eval_output_dir.mkdir(exist_ok=True)

    csv_path = eval_output_dir / "ragas_benchmark.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("📈 RAGAS EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(result)
    print("=" * 60)
    print(f"\n Detailed report saved to: {csv_path}")


if __name__ == "__main__":
    run_ragas_evaluation()
