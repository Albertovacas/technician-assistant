# Technician Assistant

A user-triggered, **multi-agent** GenAI assistant that answers pool technicians' questions from
Fluidra product manuals: grounded in the manual, with **page citations**, in the **technician's
language**, and saying *"not in the manual"* instead of guessing.

**Architecture, evaluation and production deployment:
[technician_assistant_architecture.pdf](technician_assistant_architecture.pdf)**

```
$ uv run pool-assistant ask "La bomba hace ruido, ¿qué debo revisar?"
Si la bomba hace ruido, siga estos pasos:
*   **Fijación incorrecta de la bomba:** Asegúrese de fijar correctamente la bomba [p. 33].

Sources: victoria_plus_silent_manual.pdf p. 33
Verification: supported
— lang=es · strategy=same_language · evidence: p.33, p.33, p.33 · gemini-3.1-flash-lite · ...
```

## How it works

```
                          ┌─ out of scope ─► fixed answer (1 LLM call)
question ─► TRIAGE ───────┤
                          └─ in scope ─► RETRIEVAL ─► ANSWER ─► VERIFIER ─► cited answer
                                         search,      only from   every claim     (4 calls)
                                         keep useful  kept        supported?
                                         excerpts,    excerpts    else 1 retry,
                                         rewrite once             then safe fallback
```

| Agent | Responsibility |
|---|---|
| **Triage** | In scope? language, intent, a search query in the manual's vocabulary |
| **Retrieval** | Vector search with a soft language filter, cross-encoder reranking, grading of excerpts, one query rewrite |
| **Answer** | Cited answer from the graded evidence only, safety warnings first |
| **Verifier** | Checks every claim against the evidence text; never shows unverified claims |

Orchestrated with **LangGraph** as a deterministic graph, so cost is bounded and the flow is testable.

Ingestion is template-agnostic: text layer plus structural signals per page, tables rebuilt
from the PDF's vector geometry, a vision LLM only for figures and scans (cached), language
detected per chunk, local embeddings (`embeddinggemma-300m`) and Qdrant.

## Quickstart

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). A free
[Google AI Studio](https://aistudio.google.com/apikey) key is needed to ask questions (not to
ingest).

```bash
uv sync                                        # install (creates .venv from uv.lock)
cp .env.example .env                           # then set GOOGLE_API_KEY=...

uv run pool-assistant ingest data/manuals/     # parse, chunk and index the sample manual
                                               # (~1.2 GB embedding model downloaded once)
uv run pool-assistant ask "How often must the mechanical seal be replaced?"
                                               # (~1.1 GB reranker model downloaded once)
```

Ingestion works without an API key: vision transcriptions of figure pages are committed in
`data/vision_cache/`. The index is local (embedded Qdrant in `data/index/`), no server needed.
Models are cached in `data/.cache/`; if the first-time download is interrupted by a network
error, run the command again: it resumes the download. To skip the reranker (smaller download,
faster answers, slightly lower retrieval quality), set `RERANK_MODEL=` empty in `.env`.

## Usage

```bash
uv run pool-assistant ingest data/manuals/                  # every PDF in a folder
uv run pool-assistant ingest manual.pdf --lang es           # declared language
uv run pool-assistant ask "No arranca" --trace              # show each agent step
uv run pool-assistant ask "Pump noise" --mode simple        # single-pass baseline
uv run pool-assistant ask "¿Precio de la bomba?" --json     # full Answer as JSON
LOG_FORMAT=json uv run pool-assistant ask "..."             # structured logs (stderr)
```

`--json` returns the full `Answer`: text, citations, flags (`insufficient_evidence`,
`out_of_scope`, `degraded`), verification, token usage per agent, `request_id` and the agent
trace.

## Tests and evaluation

```bash
uv run pytest                      # all tests
uv run python eval/run_eval.py     # golden set (31 questions): baseline vs multi-agent
```

The evaluation calls Gemini, so it needs `GOOGLE_API_KEY` and an ingested index. On the free
tier (15 requests/min) the full run takes several minutes. It writes a markdown report with
the metrics per mode to `eval/reports/`.

## Configuration

All settings come from environment variables / `.env` (see [.env.example](.env.example)).

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini (Google AI Studio) |
| `LLM_MODEL` | `gemini-3.1-flash-lite` | Model for all agents |
| `LLM_REQUESTS_PER_MINUTE` | `14` | Shared client-side rate limit (free tier: 15/min) |
| `AGENT_TIME_BUDGET_S` | `45` | After this, optional steps (rewrite, answer retry) are skipped |
| `QDRANT_URL` | empty (local) | Set to use a Qdrant server / Qdrant Cloud |
| `EMBEDDING_MODEL` | `google/embeddinggemma-300m` | Local ONNX model (FastEmbed) |
| `RERANK_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | Local cross-encoder; empty disables reranking |
| `RERANK_CANDIDATES` | `20` | Vector candidates reranked down to the top 5 |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | `json` for one object per line |

## Repository layout

```
src/pool_assistant/
  ingestion/    loader, tables, vision, chunker, language, indexer, pipeline
  retrieval/    retriever (soft language filter), reranker (cross-encoder)
  rag/          single-pass baseline + shared answer generation, prompts
  agents/       triage, retrieval, answer, verifier, graph, state, prompts
  evaluation/   golden set, metrics, runner, report
  cli.py · llm.py · embeddings.py · vectorstore.py · observability.py · config.py
tests/          unit (fakes, no network) · integration (sample manual)
eval/           golden_set.json, run_eval.py (writes a markdown report to eval/reports/)
data/           manuals/, vision_cache/
```