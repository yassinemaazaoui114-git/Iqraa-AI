"""
Qaloon-specific Arabic text normalisation.

The KFGQPC (King Fahd Glorious Quran Printing Complex) digital text format
uses Unicode characters beyond standard tashkeel to encode visual/typesetting
annotations — small high stops, rounded zeros, dotless heads, and similar
marks that appear in the printed mushaf but carry zero phonemic information.

This module removes those marks while preserving every diacritic (tashkeel)
that the CTC model must learn to predict.  Stripping tashkeel would make the
problem trivially wrong — the entire purpose of Iqraa AI is to detect
diacritical errors in Qaloon recitation.

Canonical usage:
    from src.normalize_text import normalize_qalon_text
    clean = normalize_qalon_text(raw_ayah_string)
"""

import re
import unicodedata


# ── Removal set ───────────────────────────────────────────────────────────────
# Built from explicit codepoint ranges using chr() to avoid escape-sequence
# ambiguity across editors and platforms.

_REMOVE_CHARS: frozenset[str] = frozenset(
    # U+06D6..U+06DC — Quranic annotation marks (small high ligatures/forms)
    [chr(c) for c in range(0x06D6, 0x06DD)]
    # U+06DF..U+06E4 — more Quranic marks (rounded zeros, dotless head, madda)
    + [chr(c) for c in range(0x06DF, 0x06E5)]
    # U+06E7, U+06E8 — small high yeh / small high noon
    + [chr(0x06E7), chr(0x06E8)]
    # U+06EA..U+06ED — low stops and small low meem
    + [chr(c) for c in range(0x06EA, 0x06EE)]
    # U+0660..U+0669 — Arabic-Indic digits (ayah-number glyphs appended to each value)
    + [chr(c) for c in range(0x0660, 0x066A)]
    # U+0620 — KFGQPC special letter mark used as a typesetting placeholder
    + [chr(0x0620)]
    # U+00A0 — non-breaking space appended before the ayah numeral in KFGQPC values
    + [chr(0x00A0)]
)

# Compiled whitespace-collapse pattern (used after character removal)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_qalon_text(text: str) -> str:
    """Normalise a single KFGQPC Qaloon ayah for use as a CTC training label.

    Processing steps (in order):
      1. NFC Unicode normalisation — canonical composition so identical-looking
         characters always have the same byte sequence.
      2. Remove KFGQPC structural marks, Arabic-Indic digit characters,
         and the non-breaking space suffix.
      3. Collapse internal whitespace to a single space and strip edges.

    What is preserved:
      All standard Arabic base letters (U+0621–U+063A, U+0641–U+064A) and
      all tashkeel diacritics (U+064B–U+0652, U+0670) pass through unchanged.

    Parameters
    ----------
    text : str
        Raw Arabic string from qalon_canonical.json.

    Returns
    -------
    str
        Clean, diacritised Arabic ready for character-level CTC tokenisation.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in _REMOVE_CHARS)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def normalize_corpus(corpus: dict[str, str]) -> dict[str, str]:
    """Normalise every ayah in a key→text corpus dict.

    Parameters
    ----------
    corpus : dict[str, str]
        Mapping of ayah keys (e.g. ``'001001'``) to raw Arabic strings,
        as loaded directly from ``qalon_canonical.json``.

    Returns
    -------
    dict[str, str]
        New dict with the same keys and normalised text values.
    """
    return {key: normalize_qalon_text(text) for key, text in corpus.items()}
