"""Command-line interface.

    pool-assistant ingest data/manuals/                 # every PDF in a folder
    pool-assistant ingest manual.pdf --lang es          # declared language
    pool-assistant ingest manual.pdf --no-vision        # text + tables only
    pool-assistant ask "La bomba hace ruido"            # multi-agent answer with citations
    pool-assistant ask "La bomba hace ruido" --trace    # + each agent step
    pool-assistant ask "Pump noise" --mode simple       # single-pass baseline
    pool-assistant ask "Pump noise" --json              # full Answer as JSON

Logs go to stderr; results go to stdout.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pool_assistant.agents.graph import MultiAgentRAG
from pool_assistant.config import Settings, get_settings
from pool_assistant.embeddings import Embedder, get_embedder
from pool_assistant.ingestion.language import normalize_lang_code
from pool_assistant.ingestion.pipeline import IngestReport, ingest_pdf
from pool_assistant.llm import LLMConfigError, get_llm
from pool_assistant.logging_config import configure_logging
from pool_assistant.rag.simple_rag import SimpleRAG
from pool_assistant.retrieval.reranker import Reranker, get_reranker
from pool_assistant.retrieval.retriever import Retriever
from pool_assistant.schemas import Answer
from pool_assistant.vectorstore import QdrantStore, VectorStoreError

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_PARTIAL_FAILURE, EXIT_USAGE = 0, 1, 2


# --- Dependency builders (module-level so tests can replace them) -------------------
def build_embedder(settings: Settings) -> Embedder:
    return get_embedder(settings)


def build_store(settings: Settings, dim: int) -> QdrantStore:
    return QdrantStore.from_settings(dim, settings)


def build_vision_llm(settings: Settings):
    """Gemini if an API key is configured; otherwise None (cached transcriptions only)."""
    if settings.llm_provider != "gemini" or settings.google_api_key is None:
        logger.warning("No GOOGLE_API_KEY: vision fallback will use cached transcriptions only")
        return None
    from pool_assistant.llm import get_vision_llm

    return get_vision_llm(settings)


def build_llm(settings: Settings):
    return get_llm(settings)


def build_reranker(settings: Settings) -> Reranker | None:
    return get_reranker(settings)


# --- ingest ---------------------------------------------------------------------------
def collect_pdfs(paths: Sequence[Path]) -> list[Path]:
    """Expand folders into their PDFs (sorted, non-recursive); keep explicit files."""
    pdfs: list[Path] = []
    for path in paths:
        if path.is_dir():
            pdfs.extend(sorted(p for p in path.iterdir() if p.suffix.lower() == ".pdf"))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path)
        else:
            raise FileNotFoundError(f"Not a PDF file or folder: {path}")
    return pdfs


def format_report(report: IngestReport) -> str:
    parsers = " · ".join(f"{k} {v}" for k, v in sorted(report.parsers.items()))
    langs = " ".join(f"{k} {v}" for k, v in sorted(report.languages.items()))
    return (
        f"✔ {report.source} → doc_id={report.doc_id}\n"
        f"    pages {report.pages} ({parsers}) · chunks {report.chunks} · {report.seconds}s\n"
        f"    languages: {langs}"
    )


def run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    try:
        pdfs = collect_pdfs(args.paths)
        explicit_lang = normalize_lang_code(args.lang) if args.lang else None
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    if not pdfs:
        print("error: no PDF files found", file=sys.stderr)
        return EXIT_USAGE
    if args.doc_id and len(pdfs) > 1:
        print("error: --doc-id can only be used with a single PDF", file=sys.stderr)
        return EXIT_USAGE

    embedder = build_embedder(settings)
    vision_llm = build_vision_llm(settings) if args.vision else None
    store = build_store(settings, embedder.dim)
    failures = 0
    try:
        for pdf in pdfs:
            try:
                report = ingest_pdf(
                    pdf, settings=settings, embedder=embedder, store=store,
                    vision_llm=vision_llm, use_vision=args.vision,
                    explicit_lang=explicit_lang, doc_id=args.doc_id,
                )
                print(format_report(report))
            except Exception as error:  # one bad PDF must not stop the batch
                failures += 1
                logger.exception("Ingestion failed for %s", pdf.name)
                print(f"✘ {pdf.name}: {error}")
        location = settings.qdrant_url or settings.qdrant_path
        print(f"\nIndex: {store.count()} chunks in '{store.collection}' ({location})")
    finally:
        store.close()  # releases the embedded-Qdrant folder lock

    print(f"{len(pdfs) - failures}/{len(pdfs)} documents ingested")
    return EXIT_PARTIAL_FAILURE if failures else EXIT_OK


# --- ask ------------------------------------------------------------------------------
def format_trace(answer: Answer) -> str:
    lines = ["", "Agent trace:"]
    for i, step in enumerate(answer.agent_trace, start=1):
        calls = f"{step.llm_calls} LLM call" + ("s" if step.llm_calls != 1 else "")
        lines.append(f"  {i}. [{step.agent}/{step.node}] {step.latency_ms} ms · {calls}"
                     f" · {step.summary}")
    total_calls = sum(s.llm_calls for s in answer.agent_trace)
    lines.append(f"  total: {total_calls} LLM calls")
    return "\n".join(lines)


def format_answer(answer: Answer) -> str:
    lines = [answer.answer, ""]
    if answer.citations:
        by_source: dict[str, list[int]] = {}
        for citation in answer.citations:
            by_source.setdefault(citation.source, []).append(citation.page_number)
        sources = "; ".join(
            f"{src} p. {', '.join(map(str, pages))}" for src, pages in by_source.items()
        )
        lines.append(f"Sources: {sources}")
    if answer.invalid_citations:
        lines.append(f"⚠ Cited pages not found in the evidence: {answer.invalid_citations}")
    if answer.degraded:
        lines.append("⚠ Degraded: the language model failed; manual excerpts shown instead.")
    if answer.verification and answer.verification.status != "skipped":
        lines.append(f"Verification: {answer.verification.status}")
    retrieval = answer.retrieval
    evidence = ", ".join(f"p.{c.chunk.page_number}" for c in retrieval.chunks) if retrieval else ""
    strategy = retrieval.strategy if retrieval else "no search (out of scope)"
    if retrieval and retrieval.reranked:
        strategy += " + rerank"
    usage = answer.usage
    tokens = f" · {usage.input_tokens}+{usage.output_tokens} tokens" if usage else ""
    lines.append(
        f"— lang={answer.language} · strategy={strategy} · "
        f"evidence: {evidence or 'none'} · {answer.model} · {answer.latency_ms} ms{tokens}"
        f" · request {answer.request_id}"
    )
    return "\n".join(lines)


def run_ask(args: argparse.Namespace, settings: Settings) -> int:
    try:
        lang = normalize_lang_code(args.lang) if args.lang else None
        llm = build_llm(settings)
    except (ValueError, LLMConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dim)
    try:
        try:
            store.check_compatible()
        except VectorStoreError as error:  # index built with another embedding model
            print(f"error: {error}", file=sys.stderr)
            return EXIT_USAGE
        if store.count() == 0:
            print("error: the index is empty. Run `pool-assistant ingest data/manuals/` first.",
                  file=sys.stderr)
            return EXIT_PARTIAL_FAILURE
        retriever = Retriever(embedder, store, k=args.k or settings.retrieval_k,
                              fallback_lang=settings.retrieval_fallback_lang,
                              reranker=build_reranker(settings),
                              candidates=settings.rerank_candidates)
        if args.mode == "agents":
            rag = MultiAgentRAG(retriever, llm, model_name=settings.llm_model,
                                time_budget_s=settings.agent_time_budget_s)
        else:
            rag = SimpleRAG(retriever, llm, model_name=settings.llm_model)
        try:
            answer = rag.answer(args.question, lang=lang)
        except Exception as error:  # provider errors (503, quota...) -> clear message
            logger.debug("LLM call failed", exc_info=True)
            print(f"error: the language model could not answer: {error}", file=sys.stderr)
            return EXIT_PARTIAL_FAILURE
    finally:
        store.close()

    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        print(format_answer(answer) + (format_trace(answer) if args.trace else ""))
    return EXIT_OK


# --- entry point ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pool-assistant", description=__doc__.split("\n")[0])
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING or ERROR")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Parse, chunk and index PDF manuals")
    ingest.add_argument("paths", nargs="+", type=Path, help="PDF files and/or folders")
    ingest.add_argument("--lang", help="ISO 639-1 language of the documents (skips detection)")
    ingest.add_argument("--doc-id", help="Document id (single PDF only; default: file name)")
    ingest.add_argument("--no-vision", dest="vision", action="store_false",
                        help="Disable the vision fallback for figures and scanned pages")
    ingest.set_defaults(handler=run_ingest)

    ask = commands.add_parser("ask", help="Answer a technician's question from the manuals")
    ask.add_argument("question", help="The question, in any language")
    ask.add_argument("--lang", help="ISO 639-1 language of the question (skips detection)")
    ask.add_argument("--k", type=int, default=None, help="Number of evidence chunks (default 5)")
    ask.add_argument("--mode", choices=["agents", "simple"], default="agents",
                     help="agents: multi-agent graph (default); simple: single-pass baseline")
    ask.add_argument("--trace", action="store_true", help="Show each agent step")
    ask.add_argument("--json", action="store_true", help="Print the full Answer as JSON")
    ask.set_defaults(handler=run_ask)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(args.log_level or settings.log_level, settings.log_format)
    return args.handler(args, settings)


if __name__ == "__main__":
    sys.exit(main())
