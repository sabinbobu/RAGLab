import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from raglab.config import settings
from raglab.gateway import LLMResponse
from raglab.gateway.factory import get_provider
from raglab.prompts import load_prompt
from raglab.retrieval.base import RetrievedChunk
from raglab.retrieval.chroma import ChromaRetriever

app = FastAPI(title="RAGLab")


class GenerateRequest(BaseModel):
    provider: str
    model: str = settings.default_model
    prompt: str


class QueryRequest(BaseModel):
    question: str
    provider: str = "openai"
    model: str = settings.default_model
    prompt_version: str = "v1"
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    cost_usd: float
    latency_ms: float


@app.post("/generate", response_model=LLMResponse)
def generate(request: GenerateRequest) -> LLMResponse:
    provider = get_provider(request.provider)
    return provider.generate(request.prompt, request.model)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    # step 1: retrieve relevant chunks from ChromaDB
    retriever = ChromaRetriever(collection_name="raglab")
    chunks = retriever.retrieve(request.question, top_k=request.top_k)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # step 2: load prompt template and inject context + question
    prompt_template = load_prompt(request.prompt_version)
    context = "\n\n".join(
        f"[{chunk.source}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
    )
    prompt = prompt_template["user"].format(
        context=context,
        question=request.question,
    )

    # step 3: call LLM via gateway with system prompt
    full_prompt = f"{prompt_template['system']}\n\n{prompt}"
    provider = get_provider(request.provider)
    llm_response = provider.generate(full_prompt, request.model)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return QueryResponse(
        answer=llm_response.text,
        sources=chunks,
        cost_usd=llm_response.cost_usd,
        latency_ms=latency_ms,
    )
