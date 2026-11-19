"""Language resolution for chunks of manual text.

Priority chain (first that applies wins):
1. explicit  - language declared at ingestion time (e.g. `ingest --lang es`, or a
               document-management metadata field). Most reliable source.
2. detected  - statistical detection with line-level voting (see below).
3. None      - not enough evidence, or genuinely multilingual text.

Why line-level voting: a single detector call always returns *some* language, and its
score is only as good as the model's calibration (lingua returned 1.0 on mixed pages;
fastText returns low scores there). Voting per line and measuring the share of
characters in the dominant language gives a detector-independent, interpretable
signal ("what fraction of this text is in one language?").
On the sample manual: single-language pages >= 0.85, multilingual pages <= 0.50.

Detector: fast-langdetect (fastText lid.176, 176 languages, ~1 MB bundled model,
offline, deterministic). Chosen over lingua (same accuracy here, 307 MB) and
langdetect (slower, non-deterministic, noisy on mixed text).
"""

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from fast_langdetect import LangDetectConfig, LangDetector

DEFAULT_MIN_SHARE = 0.7        # dominant language must cover >= 70% of the voted chars
DEFAULT_MIN_LINE_CHARS = 15    # shorter lines (labels, page numbers) are too noisy to vote
_ISO_639_1 = re.compile(r"^[a-z]{2}$")

Method = Literal["explicit", "detected", "undetermined"]


@dataclass(frozen=True)
class LanguageResult:
    lang: str | None       # ISO 639-1 code, or None
    confidence: float      # explicit: 1.0; detected: dominant-language share (0-1)
    method: Method


@lru_cache(maxsize=1)
def _detector() -> LangDetector:
    # model="lite" uses the bundled model (never downloads); no input truncation
    # (the library truncates to 80 chars by default).
    return LangDetector(LangDetectConfig(model="lite", max_input_length=None))


def _detect_line(line: str) -> str:
    return str(_detector().detect(line, model="lite", k=1)[0]["lang"])


def normalize_lang_code(code: str) -> str:
    """Validate and normalise an explicit language code ('ES' -> 'es')."""
    normalized = code.strip().lower()
    if not _ISO_639_1.match(normalized):
        raise ValueError(f"Expected an ISO 639-1 language code (e.g. 'en', 'es'), got {code!r}")
    return normalized


def detect_language(
    text: str,
    *,
    min_share: float = DEFAULT_MIN_SHARE,
    min_line_chars: int = DEFAULT_MIN_LINE_CHARS,
) -> LanguageResult:
    """Detect the dominant language of `text` by character-weighted line voting."""
    votes: Counter[str] = Counter()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) >= min_line_chars:
            votes[_detect_line(line)] += len(line)

    if not votes:
        return LanguageResult(lang=None, confidence=0.0, method="undetermined")

    lang, weight = votes.most_common(1)[0]
    share = weight / sum(votes.values())
    if share < min_share:
        return LanguageResult(lang=None, confidence=share, method="undetermined")
    return LanguageResult(lang=lang, confidence=share, method="detected")


DEFAULT_QUERY_MIN_SCORE = 0.5


def detect_query_language(text: str, min_score: float = DEFAULT_QUERY_MIN_SCORE) -> str | None:
    """Language of a short user question, or None if not confident.

    Questions are often shorter than the 15 chars line-voting needs, so this uses a single
    detector call with a confidence floor. Measured on 25 EN/ES/FR/DE questions: 23 correct;
    both errors had low scores ('Fehlercode E01' 0.17, 'Precio' 0.49) and become None.
    """
    text = " ".join(text.split())
    if len(text) < 3:
        return None
    top = _detector().detect(text, model="lite", k=1)[0]
    return str(top["lang"]) if float(top["score"]) >= min_score else None


def resolve_language(
    text: str,
    explicit: str | None = None,
    *,
    min_share: float = DEFAULT_MIN_SHARE,
) -> LanguageResult:
    """Apply the priority chain: explicit -> detected -> undetermined."""
    if explicit:
        return LanguageResult(lang=normalize_lang_code(explicit), confidence=1.0, method="explicit")
    return detect_language(text, min_share=min_share)
