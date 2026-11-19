"""Evaluation metrics, runner and report on synthetic answers (no LLM, no index)."""

from pool_assistant.evaluation.golden import GoldenItem
from pool_assistant.evaluation.metrics import by_category, score, summarize
from pool_assistant.evaluation.report import build_report
from pool_assistant.evaluation.runner import run_mode
from pool_assistant.schemas import (
    Answer,
    Citation,
    RetrievalResult,
    ScoredChunk,
    Usage,
    Verification,
)

from .test_vectorstore import make_chunk

CAPACITOR = GoldenItem(id="es-cap", question="¿Cada cuánto el condensador?", lang="es",
                       category="maintenance", expected="answer", pages=[32],
                       key_facts=[["1 año", "un año"], ["condensador"]])
PRICE = GoldenItem(id="oos-price", question="¿Precio?", lang="es", category="out_of_scope",
                   expected="out_of_scope")
FLOW = GoldenItem(id="nim-flow", question="¿Caudal en m3/h?", lang="es",
                  category="not_in_manual", expected="not_in_manual")


EVIDENCE_TEXT = "| Condensador | 1 año |"


def make_answer(text: str, *, pages_evidence=(32,), cited=(32,), invalid=(),
                evidence_text: str = EVIDENCE_TEXT, **flags) -> Answer:
    chunks = [ScoredChunk(chunk=make_chunk("m", p, 0, evidence_text, lang="es"), score=0.5)
              for p in pages_evidence]
    retrieval = RetrievalResult(question="q", query_lang="es", search_lang="es",
                                strategy="same_language", chunks=chunks)
    return Answer(question="q", answer=text, model="fake", latency_ms=1000,
                  citations=[Citation(source="m.pdf", page_number=p) for p in cited],
                  invalid_citations=list(invalid), retrieval=retrieval,
                  usage=Usage(llm_calls=4, input_tokens=3000, output_tokens=100), **flags)


def test_correct_answer() -> None:
    r = score(CAPACITOR, make_answer("El condensador debe cambiarse cada 1 año [p. 32]."), "x")

    assert r.correct and r.fact_coverage == 1.0
    assert r.evidence_hit and r.citation_hit
    assert r.lang_ok and r.tokens == 3100


def test_page_hit_without_the_fact_in_the_evidence_is_detected() -> None:
    """Regression of en-spec-temp: right page retrieved through a chunk without the fact."""
    answer = make_answer("No consta en el manual.", evidence_text="Uso por niños mayores de 8 años",
                         insufficient_evidence=True)

    r = score(CAPACITOR, answer, "x")

    assert r.evidence_hit is True            # the page is there...
    assert r.evidence_has_facts is False     # ...but not the fact: a retrieval problem
    assert r.evidence_pages == [32]


def test_evidence_with_the_facts() -> None:
    r = score(CAPACITOR, make_answer("El condensador, cada 1 año [p. 32].",
                                     evidence_text="| Condensador | 1 año |"), "x")

    assert r.evidence_has_facts is True


def test_wrong_value_misses_a_key_fact() -> None:
    r = score(CAPACITOR, make_answer("El condensador debe cambiarse cada 5 años [p. 32]."), "x")

    assert r.behavior_ok and not r.correct
    assert r.fact_coverage == 0.5 and r.missing_facts == ["1 año / un año"]


def test_wrong_page_and_invalid_citations_are_reported() -> None:
    answer = make_answer("El condensador, cada 1 año [p. 12].", pages_evidence=(12,),
                         cited=(12,), invalid=(99,))

    r = score(CAPACITOR, answer, "x")

    assert r.correct  # the facts are right...
    assert r.evidence_hit is False and r.citation_hit is False  # ...but from the wrong page
    assert r.invalid_citations == 1


def test_declining_is_correct_for_negative_items() -> None:
    declined = make_answer("Esta información no consta en el manual.", insufficient_evidence=True)
    rejected = make_answer("Solo puedo ayudar con preguntas técnicas.", out_of_scope=True)

    assert score(FLOW, declined, "x").correct
    assert score(PRICE, declined, "x").correct           # safe, though not identified as such
    assert score(PRICE, rejected, "x").behavior == "out_of_scope"


def test_inventing_an_answer_to_a_negative_item_is_wrong() -> None:
    r = score(FLOW, make_answer("El caudal máximo es de 12 m3/h [p. 32]."), "x")

    assert not r.correct and r.behavior == "answer"


def test_unverified_fallback_counts_as_declined_not_as_answered() -> None:
    fallback = make_answer("No he podido verificar una respuesta con el manual: p. 32.",
                           verification=Verification(status="unsupported"))

    assert not score(CAPACITOR, fallback, "x").correct
    assert score(FLOW, fallback, "x").correct


def test_answer_in_another_language_is_flagged() -> None:
    r = score(CAPACITOR, make_answer("The capacitor must be replaced every 1 year [p. 32]."), "x")

    assert r.answer_lang == "en" and r.lang_ok is False


def test_runner_records_errors_and_continues() -> None:
    class Flaky:
        def answer(self, question, *, lang=None):
            if "Precio" in question:
                raise RuntimeError("503")
            return make_answer("El condensador, cada 1 año [p. 32].")

    results = run_mode(Flaky(), [CAPACITOR, PRICE], "x")

    assert [r.behavior for r in results] == ["answer", "error"]
    assert results[1].error == "RuntimeError: 503"


def test_summary_and_report() -> None:
    results = {
        "simple": [score(CAPACITOR, make_answer("Cada 5 años [p. 32]."), "simple"),
                   score(PRICE, make_answer("No consta.", insufficient_evidence=True), "simple")],
        "agents": [score(CAPACITOR, make_answer("El condensador, cada 1 año [p. 32]."), "agents"),
                   score(PRICE, make_answer("Fuera de ámbito.", out_of_scope=True), "agents")],
    }

    simple, agents = summarize(results["simple"]), summarize(results["agents"])
    assert (simple["correct"], agents["correct"]) == (1, 2)
    assert simple["out_of_scope_detected"] == 0 and agents["out_of_scope_detected"] == 1
    assert by_category(results["agents"])["maintenance"] == {"items": 1, "correct": 1}

    report = build_report(results, {"model": "fake"})
    assert "| Correct (all items) | 50% | 100% |" in report
    assert "## Failures: simple (1)" in report and "es-cap" in report
