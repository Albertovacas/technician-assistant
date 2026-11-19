"""Evaluate the baseline and the multi-agent system on the golden set.

    uv run python eval/run_eval.py                      # both modes, all questions
    uv run python eval/run_eval.py --modes agents       # one mode
    uv run python eval/run_eval.py --ids es-ts-noise oos-price

Writes eval/reports/<timestamp>.md (report) and eval/results/<timestamp>.json (raw).
Needs the local index (`pool-assistant ingest data/manuals/`) and GOOGLE_API_KEY.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from pool_assistant.agents.graph import MultiAgentRAG
from pool_assistant.agents.prompts import AGENT_PROMPTS_VERSION
from pool_assistant.config import get_settings
from pool_assistant.embeddings import get_embedder
from pool_assistant.evaluation.golden import load_golden_set
from pool_assistant.evaluation.metrics import summarize
from pool_assistant.evaluation.report import build_report
from pool_assistant.evaluation.runner import run_mode
from pool_assistant.llm import get_llm
from pool_assistant.logging_config import configure_logging
from pool_assistant.rag.prompts import PROMPT_VERSION
from pool_assistant.rag.simple_rag import SimpleRAG
from pool_assistant.retrieval.reranker import get_reranker
from pool_assistant.retrieval.retriever import Retriever
from pool_assistant.vectorstore import QdrantStore

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modes", nargs="+", default=["simple", "agents"],
                        choices=["simple", "agents"])
    parser.add_argument("--ids", nargs="*", help="Only these golden ids")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_format)
    golden = load_golden_set()
    items = [i for i in golden.items if not args.ids or i.id in args.ids]

    llm, embedder = get_llm(settings), get_embedder(settings)
    store = QdrantStore.from_settings(embedder.dim, settings)
    try:
        if store.count() == 0:
            raise SystemExit("Empty index: run `pool-assistant ingest data/manuals/` first")
        retriever = Retriever(embedder, store, k=settings.retrieval_k,
                              fallback_lang=settings.retrieval_fallback_lang,
                              reranker=get_reranker(settings),
                              candidates=settings.rerank_candidates)
        pipelines = {
            "simple": SimpleRAG(retriever, llm, model_name=settings.llm_model),
            "agents": MultiAgentRAG(retriever, llm, model_name=settings.llm_model,
                                    time_budget_s=settings.agent_time_budget_s),
        }
        results = {}
        for mode in args.modes:
            print(f"== {mode}: {len(items)} questions")
            results[mode] = run_mode(
                pipelines[mode], items, mode,
                on_result=lambda r: print(f"  {'✔' if r.correct else '✘'} {r.id:22} "
                                          f"{r.behavior:13} {r.latency_ms / 1000:5.1f}s"),
            )
    finally:
        store.close()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"date": stamp, "model": settings.llm_model, "golden_set_version": golden.version,
            "questions": len(items), "simple_prompt": PROMPT_VERSION,
            "agent_prompts": AGENT_PROMPTS_VERSION, "retrieval_k": settings.retrieval_k,
            "embedding_model": settings.embedding_model,
            "reranker": (f"{settings.rerank_model} (top-{settings.rerank_candidates})"
                         if settings.rerank_model else "none")}
    (EVAL_DIR / "reports").mkdir(exist_ok=True)
    (EVAL_DIR / "results").mkdir(exist_ok=True)
    report_path = EVAL_DIR / "reports" / f"{stamp}.md"
    report_path.write_text(build_report(results, meta), encoding="utf-8")
    (EVAL_DIR / "results" / f"{stamp}.json").write_text(json.dumps({
        "meta": meta,
        "summary": {m: summarize(rs) for m, rs in results.items()},
        "results": {m: [r.to_dict() for r in rs] for m, rs in results.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {report_path}")
    for mode, rs in results.items():
        s = summarize(rs)
        print(f"{mode:7} correct {s['correct']}/{s['items']} · median "
              f"{s['latency_median_ms'] / 1000:.1f}s · {s['mean_llm_calls']:.1f} calls")


if __name__ == "__main__":
    main()
