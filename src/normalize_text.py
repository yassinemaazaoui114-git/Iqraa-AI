"""
Qaloon-specific Arabic text normalization.

Rules applied here are specific to Riwayat Qaloon an Nafi and differ from
generic Arabic normalization pipelines (which are typically Hafs-oriented).
We preserve all tashkeel (diacritics) because diacritical accuracy is the
core evaluation target of this system.
"""

import re
import unicodedata


# Unicode ranges for Arabic diacritics (tashkeel)
_TASHKEEL = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0670"

# Characters to remove entirely (punctuation, non-Arabic, tatweel)
_REMOVE_PATTERN = re.compile(
    r"[\u0640"          # tatweel (kashida) — decorative, not phonemic
    r"\u060c\u061b\u061f"  # Arabic comma, semicolon, question mark
    r"\u06d4"           # Arabic full stop
    r"\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e"  # ASCII punctuation
    r"]",
    re.UNICODE,
)


def normalize(text: str, keep_tashkeel: bool = True) -> str:
    """
    Normalize a single Arabic string for use as a CTC training label.

    Parameters
    ----------
    text : str
        Raw Arabic text, possibly containing diacritics and punctuation.
    keep_tashkeel : bool
        If True (default), preserve all diacritics.  Set False only for
        debugging or vocabulary analysis — never for final training labels.

    Returns
    -------
    str
        Cleaned, normalized string ready to be tokenized into characters.
    """
    # NFC normalization — canonical Unicode composition
    text = unicodedata.normalize("NFC", text)

    # Remove punctuation and tatweel
    text = _REMOVE_PATTERN.sub("", text)

    if not keep_tashkeel:
        text = re.sub(f"[{_TASHKEEL}]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_corpus(ayahs: list[dict], keep_tashkeel: bool = True) -> list[dict]:
    """
    Normalize the full list of ayah dicts loaded from qalon_canonical.json.

    Each dict must have at least a "text" key.  A "text_normalized" key is
    added in-place; the original "text" is preserved for reference.
    """
    for ayah in ayahs:
        ayah["text_normalized"] = normalize(ayah["text"], keep_tashkeel=keep_tashkeel)
    return ayahs
