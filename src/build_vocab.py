"""
Build the CTC character-level vocabulary from the Qaloon text corpus.

Why character-level?
  Arabic tashkeel attaches to base letters.  A word-level vocabulary would
  require a separate token for every possible (letter + diacritic) combination
  — tens of thousands of entries — and would fail entirely on unseen
  combinations.  A character-level vocabulary keeps the output space to ~80
  tokens while giving the model full diacritical resolution.

Vocabulary layout (index assignment):
  Index 0       : [PAD]  — CTC blank token; MUST be 0 for Wav2Vec2ForCTC
  Indices 1..N-2: Arabic letters and tashkeel, ordered by descending frequency
                  in the training corpus so the most-predicted tokens get low
                  indices (minor cache-locality benefit)
  Index N-1     : |      — word delimiter (Wav2Vec2 convention for space)
  Index N       : [UNK]  — unknown character; placed last for easy debugging

Canonical usage:
    from src.build_vocab import build_vocab, load_vocab
    vocab = build_vocab(normalised_texts, output_path='data/processed/vocab.json')
"""

import json
from collections import Counter
from pathlib import Path


def build_vocab(
    normalised_texts: list[str],
    output_path: str | None = None,
) -> dict[str, int]:
    """Build a character vocabulary from a list of normalised Arabic strings.

    Strings should be produced by ``normalize_text.normalize_qalon_text()``
    before being passed here.  Spaces are not counted as characters — they map
    to the ``|`` word-delimiter token which is appended automatically.

    Parameters
    ----------
    normalised_texts : list[str]
        One string per ayah, already stripped of KFGQPC marks and digits.
    output_path : str | None
        If provided, write the vocab dict to this path as JSON.

    Returns
    -------
    dict[str, int]
        Mapping from token string to integer index.
        ``vocab['[PAD]'] == 0`` is guaranteed.
        ``'|'`` and ``'[UNK]'`` are always present at the two highest indices.
    """
    char_counts: Counter[str] = Counter()
    for text in normalised_texts:
        for ch in text:
            if ch != " ":       # spaces become the | token; skip here
                char_counts[ch] += 1

    # [PAD] anchored at 0 (CTC blank requirement)
    vocab: dict[str, int] = {"[PAD]": 0}

    # Remaining characters ordered by descending frequency
    for i, (ch, _) in enumerate(char_counts.most_common(), start=1):
        vocab[ch] = i

    # Word delimiter and unknown token placed at the top
    next_idx = len(vocab)
    vocab["|"]     = next_idx
    vocab["[UNK]"] = next_idx + 1

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    return vocab


def load_vocab(vocab_path: str) -> dict[str, int]:
    """Load a previously saved vocab.json from disk."""
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def vocab_size(vocab: dict[str, int]) -> int:
    """Return the number of tokens (convenience wrapper)."""
    return len(vocab)
