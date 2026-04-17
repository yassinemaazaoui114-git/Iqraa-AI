# 02 — Text Processing

**Notebook:** `notebooks/02_text_processing.ipynb`  
**Last updated:** 2026-04-17  
**Status:** Complete

---

## Purpose

Notebook 02 takes the raw text from `qalon_canonical.json` and produces three
things the training pipeline needs before it can start:

1. **A normalised, diacritised text label for every ayah** — KFGQPC structural
   marks and ayah-number glyphs removed, all tashkeel preserved
2. **A fixed protected test set** — 22 ayahs locked away before training begins
3. **A character-level vocabulary file** (`vocab.json`) — the complete set of
   tokens the CTC model is allowed to predict

---

## Inputs

| Input | Location on Google Drive | Description |
|-------|--------------------------|-------------|
| Canonical text | `IqraaAI_Dataset/text/qalon_canonical.json` | 6 214-entry dict, keys `SSSAAA` → raw KFGQPC Arabic string |

The audio files and WAV outputs from Notebook 01 are not used in this notebook.

---

## Outputs

All outputs are written to Google Drive.

| Output | Path | Description |
|--------|------|-------------|
| Vocabulary | `processed/vocab.json` | ~80 tokens: `[PAD]`=0, chars by freq, `\|`, `[UNK]` |
| Protected test | `splits/protected_test.json` | 22 fixed ayahs, never used for training |
| Training text | `splits/train_text.json` | ~90% of non-test ayahs, normalised |
| Validation text | `splits/val_text.json` | ~10% of non-test ayahs, normalised |
| Completion marker | `processed/notebook_02_complete.json` | Summary with counts |

### vocab.json schema

```json
{
  "[PAD]": 0,
  "ل": 1,
  "ا": 2,
  "...": "...",
  "|": 78,
  "[UNK]": 79
}
```

Characters are ordered by descending frequency in the training corpus.
`[PAD]` is always 0 (required by Wav2Vec2ForCTC as the CTC blank token).
`|` and `[UNK]` are always at the two highest indices.

### splits/*.json schema

```json
{
  "001001": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
  "002001": "الم",
  "...": "..."
}
```

Keys are `SSSAAA` strings (same as qalon_canonical.json).
Values are normalised Arabic strings — KFGQPC marks stripped, tashkeel kept.

---

## What Each Cell Does

| Cell | Type | What it does |
|------|------|--------------|
| 1 | Markdown | Title, three-point purpose summary, re-run safety note |
| 2 | Code | Mount Drive; define all path constants including `SPLITS_DIR` and `PROCESSED_DIR` |
| 3 | Markdown | What KFGQPC structural marks are and why they must be removed |
| 4 | Code | Load `quran` dict; check Drive for existing splits; print status |
| 5 | Markdown | The five normalisation steps, each with a reason |
| 6 | Code | Define `normalize_qalon_text`; test on 5 sample ayahs with before/after output |
| 7 | Markdown | Protected test set explained as the "real exam" analogy |
| 8 | Code | Create or reload `protected_test.json` with 22 fixed ayahs; print all entries |
| 9 | Markdown | Character-level vocab explained; special tokens table |
| 10 | Code | Count chars in non-test corpus; print full inventory with codepoints and names; build and save `vocab.json` |
| 11 | Markdown | Why 90/10 train/val split; role of validation during training |
| 12 | Code | Shuffle remaining 6 192 ayahs with seed 42; split 90/10; save `train_text.json` and `val_text.json` |
| 13 | Markdown | Summary table of all outputs; what Notebook 03 does |
| 14 | Code | Write `notebook_02_complete.json` completion marker |

---

## The Normalisation Function

The core function is `normalize_qalon_text(text: str) -> str`, implemented in
both the notebook and `src/normalize_text.py`.

### Characters removed

| Range | Name | Why removed |
|-------|------|------------|
| U+06D6–U+06DC | Quranic small high marks | Typesetting annotation only — not recited |
| U+06DF–U+06E4 | More Quranic marks (incl. U+06E1) | Same reason |
| U+06E7, U+06E8 | Small high yeh / noon | Same reason |
| U+06EA–U+06ED | Low stops, small low meem | Same reason |
| U+0660–U+0669 | Arabic-Indic digits 0–9 | Ayah-number glyph appended to each JSON value |
| U+0620 | Arabic Letter Kashmiri Ya | KFGQPC typesetting placeholder |
| U+00A0 | Non-breaking space | Separator before the ayah numeral in KFGQPC values |

### Characters kept (selection)

| Range | Content |
|-------|---------|
| U+0621–U+063A | Core Arabic letters (alef through ghain) |
| U+0641–U+064A | Arabic letters (fa through ya) |
| U+064B–U+0652 | All tashkeel: tanwin fath/damm/kasr, fatha, damma, kasra, shadda, sukun |
| U+0670 | Superscript alef (used in words like اللَّه) |

---

## The Protected Test Set (22 Ayahs)

These 22 keys are **fixed for the lifetime of this project**:

```
001001–001007  Al-Fatihah (all 7 ayahs)
002255         Ayat al-Kursi
112001–112004  Al-Ikhlas (all 4 ayahs)
078001         An-Naba 1
080001         Abasa 1
087001         Al-Ala 1
091007         Ash-Shams 7
092005         Al-Layl 5
093006         Ad-Duha 6
097003         Al-Qadr 3
099001         Az-Zalzalah 1
101004         Al-Qariah 4
103002         Al-Asr 2
```

**Do not change these keys.** Any change invalidates comparison with previous
evaluation runs.

---

## Vocabulary Layout

```
Index 0     : [PAD]    — CTC blank (must be 0)
Indices 1..N-2 : Arabic chars, ordered by descending frequency
Index N-1   : |        — word delimiter
Index N     : [UNK]    — unknown (last, for easy debugging)
```

Expected vocabulary size: **~80 tokens** (varies slightly depending on
whether rare diacritics appear in non-test ayahs).

---

## Key Technical Decisions

### Remove KFGQPC marks, not standard tashkeel
Most Arabic NLP pipelines strip all diacritics for tasks like sentiment
analysis or machine translation. We do the opposite — we keep every diacritic
and only strip the non-phonemic KFGQPC structural marks. This is the entire
design point of the system.

### Vocabulary from non-test ayahs only
We build the vocabulary before splitting train/val, using all 6 192 non-test
ayahs. This ensures a character that only appears in the validation portion
is still in the vocabulary and can be predicted. Using only the 90% train split
for vocab construction would risk OOV characters in validation.

### Fixed protected test set
The 22 ayahs were chosen to include the most-recited passages (Al-Fatihah,
Ayat al-Kursi, Al-Ikhlas) plus a spread of shorter surahs. This ensures the
test evaluation covers both high-frequency and low-frequency recitation contexts.

### Seed 42 for reproducible shuffle
The train/val split must produce identical assignments on every run so that
validation WER numbers are comparable across training sessions.

---

## Source Module Locations

| Function | Module | Description |
|----------|--------|-------------|
| `normalize_qalon_text` | `src/normalize_text.py` | Single-ayah normalisation |
| `normalize_corpus` | `src/normalize_text.py` | Normalise a full key→text dict |
| `build_vocab` | `src/build_vocab.py` | Build vocab dict from text list |
| `load_vocab` | `src/build_vocab.py` | Load vocab.json from disk |

---

## How to Run

1. Complete Notebook 01 first (produces the WAV files and pairing manifest)
2. Open `notebooks/02_text_processing.ipynb` in Colab
3. Runtime → Run all
4. Expected output:

```
Loaded 6214 ayahs from qalon_canonical.json
Created protected test set (22 ayahs)
Unique characters found: ~75
Vocabulary size: ~80 tokens
Train split :  5572 ayahs  (90.0%)
Val split   :   620 ayahs  (10.0%)
Notebook 02 complete.
```

---

## Next Step

Run **Notebook 03 — Alignment** (`notebooks/03_alignment.ipynb`).
