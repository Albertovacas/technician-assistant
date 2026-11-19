import pytest

from pool_assistant.ingestion.language import (
    detect_language,
    normalize_lang_code,
    resolve_language,
)

SAMPLES = {
    "en": "Disconnect the unit from the power supply before any maintenance operation.\n"
          "Check that the mechanical parts are tightly secured.",
    "es": "Desconecte el aparato de la red eléctrica antes de cualquier operación de "
          "mantenimiento.\nCompruebe que las piezas mecánicas están bien sujetas.",
    "de": "Trennen Sie das Gerät vor jeder Wartungsarbeit vom Stromnetz.\n"
          "Prüfen Sie, ob die mechanischen Teile fest angezogen sind.",
    "ru": "Перед любым техническим обслуживанием отключите устройство от сети.\n"
          "Проверьте надёжность крепления механических частей.",
    "ar": "افصل الجهاز عن مصدر الطاقة قبل أي عملية صيانة.\n"
          "تحقق من أن الأجزاء الميكانيكية مثبتة بإحكام.",
}


@pytest.mark.parametrize(("lang", "text"), SAMPLES.items())
def test_detects_single_language_text(lang: str, text: str) -> None:
    result = detect_language(text)

    assert result.lang == lang
    assert result.method == "detected"
    assert result.confidence >= 0.7


def test_mixed_language_text_is_undetermined() -> None:
    mixed = "\n".join(SAMPLES[lang] for lang in ("en", "es", "de", "ru"))

    result = detect_language(mixed)

    assert result.lang is None
    assert result.method == "undetermined"
    assert result.confidence < 0.7


def test_label_soup_without_sentences_is_undetermined() -> None:
    labels = "Fig. 5\nO-ring\nJunta tórica\nBasket\nCesto\nPrefiltro\n95"

    assert detect_language(labels).lang is None


def test_empty_text_is_undetermined() -> None:
    assert detect_language("").method == "undetermined"


def test_explicit_language_wins_over_detection() -> None:
    result = resolve_language(SAMPLES["en"], explicit="ES")

    assert (result.lang, result.method, result.confidence) == ("es", "explicit", 1.0)


def test_without_explicit_language_falls_back_to_detection() -> None:
    assert resolve_language(SAMPLES["de"]).method == "detected"


@pytest.mark.parametrize("bad", ["english", "e", "es-ES", "12", ""])
def test_invalid_explicit_codes_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_lang_code(bad)


def test_long_lines_are_not_truncated() -> None:
    """Regression: fast-langdetect truncates to 80 chars by default."""
    long_line = "x" * 100 + " " + SAMPLES["es"].replace("\n", " ")

    assert detect_language(long_line).lang == "es"
