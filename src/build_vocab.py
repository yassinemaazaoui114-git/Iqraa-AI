"""
Build the CTC character-level vocabulary from the Qaloon text corpus.

Why character-level?
  Arabic diacritics (tashkeel) attach to base letters, so a word-level
  vocabulary would explode in size and miss unseen diacritic combinations.
  A character-level vocabulary keeps the output space small (~80 tokens)
  while preserving full diacritical resolution.

Special tokens required by Wav2Vec2ForCTC:
  [PAD]  — CTC blank token (index 0)
  [UNK]  — unknown character
  |      — word boundary (space substitute; wav2vec2 convention)
"""

import json
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "|"]


def build_vocab(normalized_texts: list[str], output_path: str | None = None) -> dict:
    """
    Build a character vocabulary from a list of normalized Arabic strings.

    Parameters
    ----------
    normalized_texts : list[str]
        Strings produced by normalize_text.normalize().
    output_path : str | None
        If given, write vocab.json to this path.

    Returns
    -------
    dict
        Mapping from token string to integer index, e.g. {"[PAD]": 0, ...}.
    """
    char_counts: Counter = Counter()
    for text in normalized_texts:
        for ch in text:
            if ch == " ":
                continue  # space becomes "|" special token
            char_counts[ch] += 1

    # Sort by frequency descending for human readability; index doesn't matter
    # for CTC as long as [PAD]=0.
    chars = sorted(char_counts.keys())

    vocab = {}
    for i, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = i

    offset = len(SPECIAL_TOKENS)
    for i, ch in enumerate(chars):
        vocab[ch] = i + offset

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    return vocab


def load_vocab(vocab_path: str) -> dict:
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)
