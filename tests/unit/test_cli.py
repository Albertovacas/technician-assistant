"""CLI with injected fakes: no model download, no files written outside tmp_path."""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from pool_assistant import cli
from pool_assistant.config import Settings
from pool_assistant.llm import LLMConfigError
from pool_assistant.rag.prompts import NO_ANSWER
from pool_assistant.vectorstore import QdrantStore
from tests.fakes import FakeAgentLLM

from .test_vectorstore import make_chunk


@pytest.fixture
def store(fake_embedder) -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"), "manuals", fake_embedder.dim)


@pytest.fixture(autouse=True)
def fakes(monkeypatch, tmp_path: Path, fake_embedder, store):
    settings = Settings(_env_file=None, vision_cache_dir=tmp_path / "vision_cache")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_embedder", lambda s: fake_embedder)
    monkeypatch.setattr(cli, "build_store", lambda s, dim: store)
    monkeypatch.setattr(cli, "build_vision_llm", lambda s: None)
    monkeypatch.setattr(cli, "build_reranker", lambda s: None)  # no model download
    monkeypatch.setattr(store, "close", lambda: None)  # keep it queryable after main()


def test_ingest_folder(make_pdf, store, capsys) -> None:
    folder = make_pdf("a.pdf", ["text"]).parent
    make_pdf("b.pdf", ["text", "matrix"])

    exit_code = cli.main(["ingest", str(folder), "--no-vision"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "✔ a.pdf → doc_id=a" in out and "✔ b.pdf → doc_id=b" in out
    assert "2/2 documents ingested" in out
    assert store.count(doc_id="a") > 0 and store.count(doc_id="b") > 0


def test_one_bad_pdf_does_not_stop_the_batch(make_pdf, tmp_path: Path, capsys) -> None:
    good = make_pdf("good.pdf", ["text"])
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is not a pdf")

    exit_code = cli.main(["ingest", str(good), str(bad)])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_PARTIAL_FAILURE
    assert "✔ good.pdf" in out and "✘ broken.pdf" in out
    assert "1/2 documents ingested" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["ingest", "does/not/exist.pdf"],
        ["ingest", "README.md"],
    ],
)
def test_invalid_paths_are_usage_errors(argv) -> None:
    assert cli.main(argv) == cli.EXIT_USAGE


def test_invalid_language_is_a_usage_error(make_pdf) -> None:
    assert cli.main(["ingest", str(make_pdf("a.pdf", ["text"])), "--lang", "spanish"]) == 2


def test_doc_id_requires_single_pdf(make_pdf) -> None:
    folder = make_pdf("a.pdf", ["text"]).parent
    make_pdf("b.pdf", ["text"])

    assert cli.main(["ingest", str(folder), "--doc-id", "x"]) == cli.EXIT_USAGE


def test_language_flag_is_applied(make_pdf, store) -> None:
    cli.main(["ingest", str(make_pdf("a.pdf", ["text"])), "--lang", "ES"])

    assert store.count(lang="es") == store.count() > 0


# --- ask ------------------------------------------------------------------------------
@pytest.fixture
def indexed_store(store, fake_embedder) -> QdrantStore:
    chunks = [
        make_chunk("manual", 33, 0, "3. La bomba hace ruido → Fijación incorrecta de la bomba",
                   lang="es"),
        make_chunk("manual", 13, 0, "3. The pump makes a noise → Pump incorrectly secured",
                   lang="en"),
    ]
    vectors = fake_embedder.embed_documents([c.text for c in chunks])
    store.replace_document("manual", chunks, vectors)
    return store


def use_llm(monkeypatch, *replies: str) -> None:
    monkeypatch.setattr(cli, "build_llm", lambda s: FakeListChatModel(responses=list(replies)))


def test_ask_prints_answer_sources_and_trace(monkeypatch, indexed_store, capsys) -> None:
    use_llm(monkeypatch, "Fija correctamente la bomba [p. 33].")

    exit_code = cli.main(["ask", "La bomba hace ruido, ¿qué reviso?", "--mode", "simple"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "Fija correctamente la bomba [p. 33]." in out
    assert "Sources: manual.pdf p. 33" in out
    assert "lang=es · strategy=same_language" in out


def test_ask_json_output(monkeypatch, indexed_store, capsys) -> None:
    use_llm(monkeypatch, "Fija la bomba [p. 33].")

    cli.main(["ask", "La bomba hace ruido, ¿qué reviso?", "--json", "--mode", "simple"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["citations"] == [{"source": "manual.pdf", "page_number": 33}]
    assert payload["retrieval"]["strategy"] == "same_language"
    assert payload["insufficient_evidence"] is False


def test_ask_flags_invalid_citations(monkeypatch, indexed_store, capsys) -> None:
    use_llm(monkeypatch, "Fija la bomba [p. 33] y cambia el condensador [p. 99].")

    cli.main(["ask", "La bomba hace ruido, ¿qué reviso?", "--mode", "simple"])

    assert "Cited pages not found in the evidence: [99]" in capsys.readouterr().out


def test_ask_not_in_manual(monkeypatch, indexed_store, capsys) -> None:
    use_llm(monkeypatch, NO_ANSWER)

    assert cli.main(["ask", "¿Cuál es el precio de la bomba?", "--mode", "simple"]) == cli.EXIT_OK
    assert "no consta en el manual" in capsys.readouterr().out


def test_ask_with_empty_index_explains_what_to_do(monkeypatch, capsys) -> None:
    use_llm(monkeypatch, "unused")

    assert cli.main(["ask", "La bomba hace ruido"]) == cli.EXIT_PARTIAL_FAILURE
    assert "pool-assistant ingest" in capsys.readouterr().err


def test_ask_with_index_from_another_embedding_model(monkeypatch, indexed_store, capsys) -> None:
    """Regression: a changed EMBEDDING_MODEL surfaced as 'the language model could not answer:
    shapes not aligned' instead of explaining that the index must be rebuilt."""
    use_llm(monkeypatch, "unused")
    other_model_store = QdrantStore(indexed_store.client, "manuals", indexed_store.dim + 1)
    monkeypatch.setattr(cli, "build_store", lambda s, dim: other_model_store)
    monkeypatch.setattr(other_model_store, "close", lambda: None)

    assert cli.main(["ask", "La bomba hace ruido"]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "embedding model changed" in err
    assert "language model" not in err


def test_ask_without_api_key_is_a_usage_error(monkeypatch, indexed_store, capsys) -> None:
    def no_key(settings):
        raise LLMConfigError("LLM_PROVIDER=gemini requires GOOGLE_API_KEY")

    monkeypatch.setattr(cli, "build_llm", no_key)

    assert cli.main(["ask", "La bomba hace ruido"]) == cli.EXIT_USAGE
    assert "GOOGLE_API_KEY" in capsys.readouterr().err


def test_ask_reports_provider_errors_cleanly(monkeypatch, indexed_store, capsys) -> None:
    class UnavailableLLM:
        def invoke(self, messages):
            raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(cli, "build_llm", lambda s: UnavailableLLM())

    assert cli.main(["ask", "La bomba hace ruido", "--mode", "simple"]) == cli.EXIT_PARTIAL_FAILURE
    assert "503 UNAVAILABLE" in capsys.readouterr().err


# --- ask, multi-agent mode (default) ---------------------------------------------------
def test_ask_agents_mode_with_trace(monkeypatch, indexed_store, capsys) -> None:
    from pool_assistant.agents.models import EvidenceGrade, TriageResult, VerificationResult

    llm = FakeAgentLLM(
        TriageResult(in_scope=True, language="es", intent="troubleshooting",
                     search_query="la bomba hace ruido", reason="avería"),
        EvidenceGrade(relevant_excerpts=[1], reason="responde"),
        "Fija correctamente la bomba [p. 33].",
        VerificationResult(supported=True, unsupported_claims=[], explanation="ok"),
    )
    monkeypatch.setattr(cli, "build_llm", lambda s: llm)

    exit_code = cli.main(["ask", "bomba ruido", "--trace"])

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "Fija correctamente la bomba [p. 33]." in out
    assert "[triage/triage]" in out and "[retrieval/grade]" in out and "[answer/answer]" in out
    assert "[verifier/verify]" in out
    assert "total: 4 LLM calls" in out


def test_ask_agents_mode_rejects_out_of_scope(monkeypatch, indexed_store, capsys) -> None:
    from pool_assistant.agents.models import TriageResult

    llm = FakeAgentLLM(TriageResult(in_scope=False, language="es", intent="other",
                                    search_query="", reason="precio"))
    monkeypatch.setattr(cli, "build_llm", lambda s: llm)

    assert cli.main(["ask", "¿Cuál es el precio?"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Solo puedo ayudar con preguntas técnicas" in out
    assert "no search (out of scope)" in out


def test_ask_agents_mode_degrades_when_the_llm_is_down(monkeypatch, indexed_store, capsys) -> None:
    # No scripted responses: every LLM call fails. Triage and grading fail open, the answer
    # degrades to the manual excerpts: the technician still gets something useful, not an error.
    monkeypatch.setattr(cli, "build_llm", lambda s: FakeAgentLLM())

    assert cli.main(["ask", "La bomba hace ruido"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "No he podido generar una respuesta" in out
    assert "[p. 33]" in out
    assert "Degraded" in out
