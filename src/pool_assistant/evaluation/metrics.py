"""Deterministic metrics: score one Answer against its golden item, then aggregate.

Why deterministic (no LLM judge as the main criterion): the same model (flash-lite)
answers, verifies and would judge, sharing its biases. Key facts checked against the
manual are reproducible, free and easy to defend.
"""

import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from pool_assistant.evaluation.golden import GoldenItem
from pool_assistant.ingestion.language import detect_query_language
from pool_assistant.schemas import Answer

DECLINED = {"not_in_manual", "out_of_scope", "unverified"}
_CITATION = re.compile(r"\[\s*p\.?[^\]]*\]", re.IGNORECASE)


@dataclass(frozen=True)
class ItemResult:
    id: str
    category: str
    lang: str
    expected: str
    mode: str
    # answer | not_in_manual | out_of_scope | unverified | degraded | error
    behavior: str
    behavior_ok: bool
    correct: bool                    # answer items: answered + every key fact; negatives: declined
    fact_coverage: float | None      # share of key facts present in the answer
    missing_facts: list[str] = field(default_factory=list)
    evidence_hit: bool | None = None  # an expected PAGE is in the evidence given to the LLM
    # Stricter: the evidence TEXT contains every key fact. A page can be retrieved through a
    # chunk that does not hold the fact (found with en-spec-temp: p.5 retrieved via c01,
    # "35 ºC" only in c00), so page-level hit alone overstates retrieval quality.
    evidence_has_facts: bool | None = None
    evidence_pages: list[int] = field(default_factory=list)
    citation_hit: bool | None = None  # an expected page is cited
    cited_pages: list[int] = field(default_factory=list)
    invalid_citations: int = 0
    answer_lang: str | None = None
    lang_ok: bool | None = None
    verification: str | None = None
    llm_calls: int = 0
    tokens: int = 0
    latency_ms: int = 0
    answer: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def observed_behavior(answer: Answer) -> str:
    if answer.out_of_scope:
        return "out_of_scope"
    if answer.insufficient_evidence:
        return "not_in_manual"
    if answer.degraded:
        return "degraded"
    if answer.verification and answer.verification.status == "unsupported":
        return "unverified"  # safe fallback: no claims, points to pages
    return "answer"


def answer_language(text: str) -> str | None:
    """Language of an answer, ignoring citations and markdown symbols."""
    clean = re.sub(r"[*#|_`>-]", " ", _CITATION.sub(" ", text))
    return detect_query_language(" ".join(clean.split())[:400])


def score(item: GoldenItem, answer: Answer, mode: str) -> ItemResult:
    behavior = observed_behavior(answer)
    if item.expected == "answer":
        behavior_ok = behavior == "answer"
    else:
        behavior_ok = behavior in DECLINED

    text = answer.answer.lower()
    evidence = answer.retrieval.chunks if answer.retrieval else []
    evidence_text = " ".join(c.chunk.text for c in evidence).lower()
    coverage, missing, evidence_has_facts = None, [], None
    if item.expected == "answer":
        missing = [" / ".join(fact) for fact in item.key_facts
                   if not any(alt.lower() in text for alt in fact)]
        coverage = 1 - len(missing) / len(item.key_facts)
        evidence_has_facts = all(any(alt.lower() in evidence_text for alt in fact)
                                 for fact in item.key_facts)

    evidence_pages = {c.chunk.page_number for c in evidence}
    cited = [c.page_number for c in answer.citations]
    expected_pages = set(item.pages)
    detected = answer_language(answer.answer)
    usage = answer.usage

    return ItemResult(
        id=item.id, category=item.category, lang=item.lang, expected=item.expected, mode=mode,
        behavior=behavior, behavior_ok=behavior_ok,
        correct=behavior_ok and (coverage is None or coverage == 1.0),
        fact_coverage=coverage, missing_facts=missing,
        evidence_hit=bool(expected_pages & evidence_pages) if expected_pages else None,
        evidence_has_facts=evidence_has_facts,
        evidence_pages=sorted(evidence_pages),
        citation_hit=bool(expected_pages & set(cited)) if expected_pages else None,
        cited_pages=cited, invalid_citations=len(answer.invalid_citations),
        answer_lang=detected, lang_ok=(detected == item.lang) if detected else None,
        verification=answer.verification.status if answer.verification else None,
        llm_calls=usage.llm_calls if usage else 0,
        tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
        latency_ms=answer.latency_ms, answer=answer.answer,
    )


def error_result(item: GoldenItem, mode: str, error: Exception) -> ItemResult:
    return ItemResult(id=item.id, category=item.category, lang=item.lang,
                      expected=item.expected, mode=mode, behavior="error", behavior_ok=False,
                      correct=False, fact_coverage=0.0 if item.expected == "answer" else None,
                      error=f"{type(error).__name__}: {error}")


# --- aggregation ------------------------------------------------------------------------------
def _rate(values: list[bool | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def _p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))] if ordered else 0


def summarize(results: list[ItemResult]) -> dict:
    """Headline metrics of one mode."""
    answers = [r for r in results if r.expected == "answer"]
    negatives = [r for r in results if r.expected != "answer"]
    oos = [r for r in results if r.expected == "out_of_scope"]
    latencies = [r.latency_ms for r in results if r.behavior != "error"]
    return {
        "items": len(results),
        "correct": sum(r.correct for r in results),
        "accuracy": _rate([r.correct for r in results]),
        "answer_items": len(answers),
        "answered": _rate([r.behavior_ok for r in answers]),
        "fully_correct_answers": _rate([r.correct for r in answers]),
        "mean_fact_coverage": statistics.mean(r.fact_coverage or 0.0 for r in answers)
        if answers else None,
        "evidence_hit": _rate([r.evidence_hit for r in answers]),
        "evidence_has_facts": _rate([r.evidence_has_facts for r in answers]),
        "citation_hit": _rate([r.citation_hit for r in answers]),
        "invalid_citations": sum(r.invalid_citations for r in results),
        "lang_ok": _rate([r.lang_ok for r in results]),
        "declined_when_expected": _rate([r.behavior_ok for r in negatives]),
        "out_of_scope_detected": _rate([r.behavior == "out_of_scope" for r in oos]),
        "verification": dict(sorted(_count(r.verification for r in results).items())),
        "errors": sum(r.behavior == "error" for r in results),
        "mean_llm_calls": statistics.mean(r.llm_calls for r in results) if results else 0,
        "mean_tokens": statistics.mean(r.tokens for r in results) if results else 0,
        "latency_median_ms": int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": _p95(latencies),
    }


def _count(values) -> dict:
    counts: dict = defaultdict(int)
    for value in values:
        if value is not None:
            counts[value] += 1
    return dict(counts)


def by_category(results: list[ItemResult]) -> dict[str, dict]:
    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
    return {cat: {"items": len(rs), "correct": sum(r.correct for r in rs)}
            for cat, rs in sorted(groups.items())}
