import os
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from bible_rag.config import (
    DEFAULT_LLM_MODEL,
    LLM_PROVIDER,
    NVIDIA_API_KEY,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
)
from bible_rag.vectorstore.retriever import BibleRetriever


class AgentState(TypedDict):
    query: str
    retrieved_verses: list[dict[str, Any]]
    response: str

SYSTEM_PROMPT = """You are an expert bilingual theological AI assistant specializing in comparing the English King James Version (KJV) and the Chinese Union Version (CUV / 和合本).

Rules:
1. Answer the user prompt clearly in the language used (or bilingually if requested).
2. Base your response STRICTLY on the retrieved Bible verses in context.
3. Every theological statement MUST be grounded with inline citations [Book Chapter:Verse].
4. Append a "Bilingual Verse Comparison" block showing CUV and KJV side-by-side or stacked.

Context:
{context}
"""

class BibleRAGAgent:
    def __init__(self, provider: str = LLM_PROVIDER, model_name: str = DEFAULT_LLM_MODEL):
        self.retriever = BibleRetriever()
        self.provider = provider.lower()
        self.model_name = model_name

        if self.provider == "ollama":
            from langchain_ollama import ChatOllama
            print(f"🤖 Initializing Local Ollama LLM [{self.model_name}] at {OLLAMA_BASE_URL}...")
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2
            )
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("❌ Missing OPENAI_API_KEY in .env file!")
            print(f"⚡ Initializing OpenAI LLM [{self.model_name}]...")
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.2, api_key=api_key)
        else: # Default: NVIDIA NIM
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            api_key = NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("❌ Missing NVIDIA_API_KEY in .env file!")
            print(f"🚀 Initializing NVIDIA NIM LLM [{self.model_name}]...")
            self.llm = ChatNVIDIA(model=self.model_name, api_key=api_key, temperature=0.2)

        self.graph = self._build_graph()

    def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        verses = self.retriever.retrieve(state["query"], top_k=3)
        return {"retrieved_verses": verses}

    def _generate_node(self, state: AgentState) -> dict[str, Any]:
        verses = state["retrieved_verses"]
        context_blocks = [
            f"Reference: {v['book_name_en']} ({v['book_name_zh']}) {v['chapter']}:{v['verse']}\nKJV: {v['text_kjv']}\nCUV: {v['text_cuv']}"
            for v in verses
        ]
        formatted_context = "\n---\n".join(context_blocks)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", "{query}")])
        chain = prompt | self.llm
        res = chain.invoke({"context": formatted_context, "query": state["query"]})
        return {"response": res.content}

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("generate", self._generate_node)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def run(self, query: str) -> str:
        final_state = self.graph.invoke({"query": query, "retrieved_verses": [], "response": ""})
        return final_state["response"]

    def close(self):
        self.retriever.close()
