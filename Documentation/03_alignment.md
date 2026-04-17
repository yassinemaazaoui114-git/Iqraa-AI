# 03 — Alignment

**Notebook:** `notebooks/03_alignment.ipynb`  
**Last updated:** 2026-04-17  
**Status:** Complete  
**Runtime required:** T4 GPU (or better)

---

## Purpose

Notebook 01 produced 114 surah-level WAV files. Surah 2 (Al-Baqarah) is
nearly two hours of audio. The model cannot train on a two-hour clip — it
needs short, labelled segments one per ayah, typically 2–15 seconds each.

This notebook cuts each surah WAV into individual ayah-level clips using
forced alignment with WhisperX.

---

## Inputs

| Input | Location on Google Drive | Description |
|-------|--------------------------|-------------|
| Surah WAV files | `processed/wav/SSS.wav` | 114 files, 16 kHz mono, −23 LUFS (from Notebook 01) |
| Canonical text | `text/qalon_canonical.json` | 6 214-entry dict from Notebook 01 |
| Pairing manifest | `processed/pairing_manifest.json` | Surah metadata from Notebook 01 |

---

## Outputs

All outputs are written to Google Drive.

| Output | Path | Description |
|--------|------|-------------|
| Ayah segments | `processed/segments/SSS_AAA.wav` | One 16 kHz WAV per ayah, ~2–15 s each |
| Per-surah checkpoints | `processed/alignment_checkpoints/SSS.json` | Segment list per surah, written immediately after each surah |
| Master manifest | `processed/segments_manifest.json` | All segments merged: keys, text labels, paths, durations |
| Completion marker | `processed/notebook_03_complete.json` | Summary with total count, duration, and any failed surahs |

### segments_manifest.json schema

```json
[
  {
    "ayah_key": "001001",
    "surah": 1,
    "ayah": 1,
    "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "wav_path": ".../segments/001_001.wav",
    "duration_sec": 3.24,
    "alignment_method": "whisperx"
  },
  ...
]
```

`alignment_method` is either `"whisperx"` (word-count matching succeeded) or
`"equal_time"` (fallback was used for this surah).

### alignment_checkpoints/SSS.json schema

Identical to `segments_manifest.json` entries but contains only the segments
for one surah. Written atomically after that surah finishes. Presence of this
file means the surah is complete and will be skipped on re-runs.

---

## What Each Cell Does

| Cell | Type | What it does |
|------|------|--------------|
| 1 | Markdown | Purpose, the WhisperX approach, checkpointing overview |
| 2 | Code | Mount Drive; define `WAV_DIR`, `SEGMENTS_DIR`, `CHECKPOINT_DIR`, `TEXT_FILE`, `MANIFEST_FILE` |
| 3 | Markdown | Why WhisperX; what word-level timestamps look like |
| 4 | Markdown | Install note; GPU requirement warning |
| 5 | Code | `!pip install -q whisperx`; `torch.cuda.is_available()` check; raises if no GPU |
| 6 | Markdown | Checkpoint format and the `alignment_method` field explained |
| 7 | Code | Inline `normalize_qalon_text`; load `quran` dict; scan checkpoint dir; report resume state |
| 8 | Markdown | What a segment is; 100 ms padding rationale |
| 9 | Markdown | What happens per surah — six steps |
| 10 | Markdown | Helper functions introduction |
| 11 | Code | `_words_to_ayah_boundaries()` + `align_surah()` definitions |
| 12 | Markdown | Main loop explanation with checkpoint skip logic |
| 13 | Code | Main loop — iterates surahs 1–114, skips completed, calls `align_surah`, writes checkpoints |
| 14 | Markdown | Build master manifest explanation |
| 15 | Code | Merge all checkpoint files into `segments_manifest.json`; print total duration |
| 16 | Markdown | What to do if surahs failed |
| 17 | Code | Quality check: segment counts vs expected, duration histogram, segments < 0.5 s, 5 random samples |
| 18 | Markdown | Summary table; what Notebook 04 will do |
| 19 | Code | Write `notebook_03_complete.json` completion marker |

---

## Alignment Approach

### Stage 1 — WhisperX transcription

`whisperx.load_model('large-v3', device='cuda', compute_type='float16', language='ar')`
is applied to each surah WAV. Whisper produces a rough word-level transcript
with approximate start/end timestamps. **The Whisper text is discarded** — we
only keep the timing. This is essential: Whisper was trained primarily on Hafs
recitation; using its transcript would produce wrong diacritics for Qaloon.

### Stage 2 — Wav2Vec2 forced alignment

`whisperx.load_align_model(language_code='ar', device='cuda')` refines the
Whisper timestamps using a Wav2Vec2 model trained for Arabic phoneme-to-audio
alignment. This step produces millisecond-precision `word_segments`:

```python
[{"word": "بسم", "start": 0.42, "end": 0.91},
 {"word": "الله", "start": 0.94, "end": 1.48}, ...]
```

### Stage 3 — Word-to-ayah boundary matching (`_words_to_ayah_boundaries`)

**Primary method — word-count matching:**

1. Count space-separated words in each normalised Qaloon ayah.
2. Assign that many consecutive WhisperX words to each ayah in sequence.
3. Ayah boundary = `[first_word.start, last_word.end]`.

**Fallback — equal-time division:**

Triggered when either:
- WhisperX produced zero words with valid timestamps, OR
- `|total_wx_words − total_qaloon_words| / total_qaloon_words > 0.35`

In fallback mode the surah duration is divided equally across its ayah count.
The `alignment_method` field is set to `"equal_time"` for all segments of
that surah.

### Stage 4 — Padding and cutting

Each boundary gets 100 ms of padding on both sides, clamped to
`[0, total_duration]`. This prevents cutting off the first or last phoneme
of an ayah, which is a common problem with hard boundary detection.

---

## Checkpointing Strategy

Processing 114 surahs takes 2–4 hours on a T4. Colab may disconnect during
this time. To handle this safely:

- A checkpoint file is written to `CHECKPOINT_DIR/SSS.json` **immediately
  after** each surah succeeds.
- On every run, Cell 7 scans the checkpoint directory and reports how many
  surahs are already done.
- The main loop skips any surah whose checkpoint file exists.
- Re-running the notebook is always safe: no segments are duplicated and no
  completed work is lost.

---

## What To Do If Surahs Fail

The most common failure causes and their solutions:

| Cause | Symptom | Fix |
|-------|---------|-----|
| GPU OOM | `CUDA out of memory` error for a long surah | Re-run the notebook; GPU memory is freed between surahs |
| Network timeout | `OSError` during Drive write | Re-run; the incomplete checkpoint is not written so the surah retries cleanly |
| WAV not found | `Surah NNN: WAV not found, skipping` | Re-run Notebook 01 for the missing surah |
| WhisperX model crash | Unrecoverable CUDA error | Factory-reset the Colab runtime, reload models (Cell 5), re-run |
| Persistent failure | Surah fails 3+ times in a row | Note the surah number; exclude it from the dataset and continue |

**To retry failed surahs:** simply re-run the notebook from Cell 12 (the main
loop). Completed surahs are skipped automatically; only the failed ones are
attempted again.

If a surah is excluded, note it in the `failed_surahs` field of
`notebook_03_complete.json`. Downstream notebooks read only from
`segments_manifest.json`, which contains only successfully aligned surahs, so
exclusion does not break anything.

---

## Key Technical Decisions

### Discard WhisperX transcription, keep only timing

Using Whisper's transcript would introduce Hafs diacritics into the training
labels. Qaloon and Hafs differ in ~10% of ayahs. We replace every Whisper
word with the corresponding Qaloon text from `qalon_canonical.json`. This is
the primary reason the aligner is a two-step pipeline rather than an
end-to-end ASR system.

### 35% mismatch threshold for fallback

A 35% divergence between WhisperX word count and Qaloon word count indicates
that the alignment is unreliable for that surah. Equal-time fallback is
preferred over a misaligned word assignment because equal-time at least
preserves proportional timing. Surahs that hit the fallback are flagged by
`alignment_method = "equal_time"` in the manifest.

### 100 ms padding per boundary

Without padding, a hard cut at the exact word boundary often clips the first
or last phoneme. 100 ms is enough to include the coarticulation on both sides
without overlapping the adjacent ayah in typical recitation.

### Per-surah checkpointing (not per-ayah)

Per-ayah checkpointing would produce 6 214 small files, making the checkpoint
directory slow to list and merge. Per-surah checkpointing produces 114 files
— manageable — while still making a 2–4 hour job safely restartable with at
most one surah of wasted work.

### Inline `normalize_qalon_text` in notebook

The notebook inlines a copy of the function rather than importing from
`src/normalize_text.py`. This is intentional: Colab does not have the local
`src/` on its Python path by default, and the function is short enough that
an inline copy is the simplest approach. If `src/normalize_text.py` is ever
updated, the inline copy in this notebook must be kept in sync.

---

## Source Module Locations

| Function | Module | Description |
|----------|--------|-------------|
| `normalize_qalon_text` | `src/normalize_text.py` | Single-ayah normalisation (inline copy in notebook) |
| `_words_to_ayah_boundaries` | Cell 11 | WhisperX word timestamps → ayah (start, end) pairs |
| `align_surah` | Cell 11 | End-to-end pipeline for one surah |

---

## How to Run

1. Complete Notebooks 01 and 02 first.
2. Open `notebooks/03_alignment.ipynb` in Colab.
3. Runtime → Change runtime type → T4 GPU.
4. Runtime → Run all.
5. Estimated time: 2–4 hours for 114 surahs.
6. Expected final output:

```
Total segments   : 6214
Total duration   : ~9.5 hours
Average duration : ~5.5 s per segment
Marker saved to  : .../processed/notebook_03_complete.json
Notebook 03 complete.
```

---

## Next Step

Run **Notebook 04 — Training** (`notebooks/04_training.ipynb`).

Notebook 04 reads `segments_manifest.json` and `vocab.json` to construct a
HuggingFace `DatasetDict`, then fine-tunes
`facebook/wav2vec2-large-xlsr-53-arabic` using CTC loss on the training split.
