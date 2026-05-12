import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from raglab.config import settings
from raglab.experiments import ExperimentConfig, run_experiment
from raglab.experiments.eval import Scorecard, evaluate_experiment
from raglab.gateway import LLMResponse
from raglab.gateway.factory import get_provider
from raglab.prompts import load_prompt
from raglab.retrieval.base import RetrievedChunk
from raglab.retrieval.chroma import ChromaRetriever
from raglab.telemetry import setup_telemetry

app = FastAPI(title="RAGLab")

setup_telemetry(app)


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
    return provider.generate(
        [{"role": "user", "content": request.prompt}], request.model
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    # step 1: retrieve relevant chunks from ChromaDB
    retriever = ChromaRetriever(collection_name="raglab")
    chunks = retriever.retrieve(request.question, top_k=request.top_k)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # step 2: load and validate prompt template
    prompt_template = load_prompt(request.prompt_version)

    # format context — each chunk prefixed with citation so LLM can reference it
    context = "\n\n".join(
        f"[{chunk.source}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
    )

    # inject context and question into user template
    user_content = prompt_template["user"].format(
        context=context,
        question=request.question,
    )

    # step 3: build messages list with proper roles
    # system and user are separate — both providers handle this correctly now
    messages = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": user_content},
    ]

    # step 4: call LLM via gateway
    provider = get_provider(request.provider)
    llm_response = provider.generate(messages, request.model)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return QueryResponse(
        answer=llm_response.text,
        sources=chunks,
        cost_usd=llm_response.cost_usd,
        latency_ms=latency_ms,
    )


@app.post("/experiments/run")
def run_experiments(config: ExperimentConfig) -> dict[str, Any]:
    results = run_experiment(config)
    return {
        "total_runs": len(results),
        "experiment_id": results[0].experiment_id if results else None,
        "runs": [r.model_dump() for r in results],
    }


@app.post("/experiments/evaluate")
def evaluate(experiment_id: str) -> list[Scorecard]:
    return evaluate_experiment(experiment_id)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "raglab"}
