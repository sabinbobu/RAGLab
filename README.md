# RAGLab

RAG evaluation platform that compares LLMs, retrieval strategies, and prompt versions side-by-side — with faithfulness, cost, and latency scorecards.

**RAGLab** — answers one question: which combination of LLM, retriever, and prompt actually performs best?

Here's what the data showed on a technical AI corpus:

- gpt-4o-mini is **10x cheaper** than claude-haiku per query
- claude-haiku is **2x faster**, but faithfulness drops from 0.97 to 0.85 when switching to a conversational prompt style
- gpt-4o-mini **stays consistent** across both prompt versions
The platform runs a full experiment matrix — models × prompts × retrievers × questions — scores each run with Ragas faithfulness, and returns a cost/latency scorecard you can actually defend to stakeholders.

**Built with:** FastAPI · ChromaDB · OpenAI · Anthropic · Ragas · Docker

No LangChain. Every abstraction built from scratch.

#genai #rag #llm #python #mlops


## Architecture

```
React UI
      │ REST
FastAPI backend
  ├── LLM Gateway        OpenAI + Anthropic (LLMProvider Protocol)
  ├── Ingestion Pipeline PDF → chunks → embeddings → ChromaDB
  ├── Retrieval          Dense (ChromaDB) · BM25 · Hybrid RRF
  ├── Prompt Registry    Versioned YAML templates
  ├── Experiment Runner  Matrix: models × prompts × questions
  └── Eval Harness       Ragas faithfulness · cost · latency
```

## Eval results

Corpus: AI orchestration technical document
3 questions · 2 prompt versions · 2 providers

| Model | Prompt | Faithfulness | Avg cost | Avg latency |
|---|---|---|---|---|
| gpt-4o-mini | v1 | 0.974 | $0.000161 | 3.32s |
| gpt-4o-mini | v2 | 0.978 | $0.000197 | 4.22s |
| claude-haiku-4-5 | v1 | 0.967 | $0.001569 | 1.80s |
| claude-haiku-4-5 | v2 | 0.852 | $0.001941 | 2.71s |

**Key findings:**
- gpt-4o-mini is 10x cheaper than claude-haiku for this corpus
- claude-haiku is 2x faster but faithfulness drops from 0.97 to 0.85 on conversational prompts
- Hybrid RRF retrieval produces the most complete answers

## Quickstart

```bash
git clone https://github.com/sabinbobu/RAGLab
cd RAGLab
uv pip install -e .
cp .env.example .env  # add your API keys

# ingest a corpus
uv run raglab ingest data/sample/

# start the API
uv run uvicorn raglab.main:app --reload

# or with Docker
docker compose up
```

## Run an experiment

```bash
curl -X POST http://localhost:8000/experiments/run \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["gpt-4o-mini", "claude-haiku-4-5-20251001"],
    "prompt_versions": ["v1", "v2"],
    "questions": ["What is AI orchestration?"],
    "provider": "openai"
  }'
```

## Evaluate an experiment

```bash
curl -X POST "http://localhost:8000/experiments/evaluate?experiment_id=<id>"
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/generate` | Single LLM completion |
| POST | `/query` | RAG query with citations |
| POST | `/experiments/run` | Run model × prompt × question matrix |
| POST | `/experiments/evaluate` | Score runs with Ragas |

## Tech stack

- **Backend:** FastAPI · Pydantic v2 · SQLModel · ChromaDB
- **LLMs:** OpenAI + Anthropic via native SDKs — no LangChain
- **Retrieval:** Dense vectors · BM25 · Hybrid RRF
- **Eval:** Ragas faithfulness metric
- **Infra:** Docker · GitHub Actions CI
- **Dev:** uv · ruff · mypy · pytest · pre-commit
