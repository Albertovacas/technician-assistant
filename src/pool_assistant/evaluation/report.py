"""Markdown report comparing modes on the golden set."""

from pool_assistant.evaluation.metrics import ItemResult, by_category, summarize

ROWS = [  # (label, key, format)
    ("Correct (all items)", "accuracy", "pct"),
    ("Answer items: answered", "answered", "pct"),
    ("Answer items: every key fact present", "fully_correct_answers", "pct"),
    ("Answer items: mean key-fact coverage", "mean_fact_coverage", "pct"),
    ("Evidence page hit (expected page given to the LLM)", "evidence_hit", "pct"),
    ("Evidence contains every key fact (stricter)", "evidence_has_facts", "pct"),
    ("Citation hit (expected page cited)", "citation_hit", "pct"),
    ("Invalid citations (total)", "invalid_citations", "int"),
    ("Answer in the question's language", "lang_ok", "pct"),
    ("Declined when expected (not in manual / out of scope)", "declined_when_expected", "pct"),
    ("Out of scope identified as such", "out_of_scope_detected", "pct"),
    ("Errors", "errors", "int"),
    ("Mean LLM calls per question", "mean_llm_calls", "float"),
    ("Mean tokens per question", "mean_tokens", "int"),
    ("Latency median (s)", "latency_median_ms", "sec"),
    ("Latency p95 (s)", "latency_p95_ms", "sec"),
]


def _fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "sec":
        return f"{value / 1000:.1f}"
    if kind == "float":
        return f"{value:.1f}"
    return str(int(value))


def build_report(results: dict[str, list[ItemResult]], meta: dict) -> str:
    modes = list(results)
    summaries = {mode: summarize(rs) for mode, rs in results.items()}
    lines = ["# Evaluation report", ""]
    lines += [f"- **{key}**: {value}" for key, value in meta.items()]
    lines += ["", "## Summary", "", "| Metric | " + " | ".join(modes) + " |",
              "|---|" + "---|" * len(modes)]
    for label, key, kind in ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(summaries[m][key], kind) for m in modes)
                     + " |")
    if any(summaries[m]["verification"] for m in modes):
        lines.append("| Verification outcomes | " + " | ".join(
            ", ".join(f"{k}: {v}" for k, v in summaries[m]["verification"].items()) or "—"
            for m in modes) + " |")

    lines += ["", "## Correct by category", "",
              "| Category | Items | " + " | ".join(modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    categories = {m: by_category(rs) for m, rs in results.items()}
    for category, stats in categories[modes[0]].items():
        lines.append(f"| {category} | {stats['items']} | " + " | ".join(
            str(categories[m][category]["correct"]) for m in modes) + " |")

    for mode in modes:
        failures = [r for r in results[mode] if not r.correct]
        lines += ["", f"## Failures: {mode} ({len(failures)})", ""]
        if not failures:
            lines.append("None.")
        for r in failures:
            detail = r.error or (f"behavior={r.behavior}, missing facts={r.missing_facts}, "
                                 f"cited={r.cited_pages}, evidence pages={r.evidence_pages}, "
                                 f"evidence has facts={r.evidence_has_facts}, "
                                 f"verification={r.verification}")
            snippet = " ".join(r.answer.split())[:220]
            lines += [f"- **{r.id}** ({r.category}, expected {r.expected}): {detail}",
                      f"  > {snippet}"]
    lines += ["", "## Notes", "",
              "- Deterministic metrics: key facts were verified against the manual "
              "(tests/integration/test_golden_set_manual.py).",
              "- The question language is NOT given to the pipelines: detection is part of "
              "the evaluation.",
              "- Small set (one manual): differences of 1-2 questions are not conclusive.",
              "- Latency includes the client-side rate limiter (free tier, 14 calls/min)."]
    return "\n".join(lines) + "\n"
