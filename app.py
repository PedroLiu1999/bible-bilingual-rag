# app.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from bible_rag.agent.rag_agent import BibleRAGAgent

agent: BibleRAGAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("🚀 Initializing Bible RAG Agent...")
    agent = BibleRAGAgent()
    yield
    print("🛑 Shutting down Bible RAG Agent...")
    if agent:
        agent.close()


app = FastAPI(title="Bilingual Bible RAG Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        res = agent.run(req.query.strip())
        return QueryResponse(response=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return HTML_CONTENT


# Embedded Modern Responsive Frontend (Tailwind CSS + Alpine.js)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bilingual Bible RAG Agent (CUV & KJV)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        .prose p { margin-bottom: 0.75rem; }
        .prose ul { list-style-type: disc; padding-left: 1.25rem; margin-bottom: 0.75rem; }
        .prose blockquote { border-left: 4px solid #3b82f6; padding-left: 1rem; color: #4b5563; font-style: italic; margin: 1rem 0; }
        .prose h1, .prose h2, .prose h3 { font-weight: 700; margin-top: 1rem; margin-bottom: 0.5rem; }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col font-sans">

    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-10 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center gap-3">
            <span class="text-2xl">📖</span>
            <div>
                <h1 class="text-lg font-bold text-slate-100">Bilingual Bible RAG</h1>
                <p class="text-xs text-slate-400">CUV (和合本) & KJV Comparative AI Agent</p>
            </div>
        </div>
        <div class="flex items-center gap-2">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                NIM / Qdrant Active
            </span>
        </div>
    </header>

    <!-- Chat Area -->
    <main x-data="chatApp()" class="flex-1 max-w-4xl w-full mx-auto p-4 flex flex-col justify-between overflow-hidden">

        <!-- Messages List -->
        <div class="flex-1 overflow-y-auto space-y-6 pb-6 pr-2" id="chat-box">

            <!-- Welcome Card -->
            <div class="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 text-center my-6">
                <h2 class="text-xl font-semibold text-slate-200 mb-2">Welcome to the Bilingual Scripture Assistant</h2>
                <p class="text-sm text-slate-400 max-w-md mx-auto mb-4">
                    Ask questions in English or Chinese. Retrieve aligned CUV and KJV verses with exact verse citations.
                </p>
                <div class="flex flex-wrap justify-center gap-2 text-xs">
                    <button @click="setInput('What does John 3:16 say?')" class="bg-slate-700/50 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-600/50 transition">
                        "What does John 3:16 say?"
                    </button>
                    <button @click="setInput('What are the teachings on peacemakers and forgiveness?')" class="bg-slate-700/50 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-600/50 transition">
                        "What are the teachings on peacemakers?"
                    </button>
                    <button @click="setInput('關於饒恕和寬恕的教導')" class="bg-slate-700/50 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-600/50 transition">
                        "關於饒恕和寬恕的教導"
                    </button>
                </div>
            </div>

            <!-- Dynamic Messages -->
            <template x-for="(msg, idx) in messages" :key="idx">
                <div :class="msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'">
                    <div :class="msg.role === 'user'
                        ? 'bg-blue-600 text-white rounded-2xl rounded-tr-none px-5 py-3.5 max-w-[80%] shadow-lg'
                        : 'bg-slate-800 border border-slate-700/70 text-slate-200 rounded-2xl rounded-tl-none px-6 py-5 max-w-[90%] shadow-md prose prose-invert max-w-none'">

                        <div x-html="renderMarkdown(msg.content)"></div>
                    </div>
                </div>
            </template>

            <!-- Loading Skeleton -->
            <div x-show="loading" class="flex justify-start">
                <div class="bg-slate-800 border border-slate-700/70 rounded-2xl rounded-tl-none px-6 py-4 flex items-center gap-3">
                    <div class="flex space-x-1.5">
                        <div class="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce"></div>
                        <div class="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                        <div class="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                    </div>
                    <span class="text-xs text-slate-400 font-medium">Searching CUV & KJV vector index...</span>
                </div>
            </div>
        </div>

        <!-- Input Bar -->
        <div class="mt-2 bg-slate-950 border border-slate-800 rounded-2xl p-2 flex items-center gap-2 shadow-xl">
            <input
                type="text"
                x-model="inputQuery"
                @keydown.enter="sendQuery()"
                placeholder="Ask a question or request a verse (e.g. John 3:16, 約翰福音 3:16)..."
                class="flex-1 bg-transparent px-4 py-2.5 text-slate-100 placeholder-slate-500 focus:outline-none text-sm"
                :disabled="loading"
            >
            <button
                @click="sendQuery()"
                :disabled="loading || !inputQuery.trim()"
                class="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium px-5 py-2.5 rounded-xl transition flex items-center gap-2 text-sm">
                <span>Send</span>
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
            </button>
        </div>
    </main>

    <script>
        function chatApp() {
            return {
                inputQuery: '',
                loading: false,
                messages: [],
                setInput(val) {
                    this.inputQuery = val;
                },
                renderMarkdown(text) {
                    return marked.parse(text);
                },
                async sendQuery() {
                    const q = this.inputQuery.trim();
                    if (!q || this.loading) return;

                    this.messages.push({ role: 'user', content: q });
                    this.inputQuery = '';
                    this.loading = true;
                    this.scrollToBottom();

                    try {
                        const res = await fetch('/api/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: q })
                        });

                        const data = await res.json();
                        if (res.ok) {
                            this.messages.push({ role: 'assistant', content: data.response });
                        } else {
                            this.messages.push({ role: 'assistant', content: '❌ Error: ' + (data.detail || 'Failed to fetch response.') });
                        }
                    } catch (err) {
                        this.messages.push({ role: 'assistant', content: '❌ Network error connecting to backend server.' });
                    } finally {
                        this.loading = false;
                        this.scrollToBottom();
                    }
                },
                scrollToBottom() {
                    setTimeout(() => {
                        const box = document.getElementById('chat-box');
                        if (box) box.scrollTop = box.scrollHeight;
                    }, 100);
                }
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
