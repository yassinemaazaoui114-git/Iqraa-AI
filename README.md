# Iqraa AI

**Quranic recitation correction for Riwayat Qaloon an Nafi**

Iqraa AI is an automatic speech recognition (ASR) system fine-tuned specifically for detecting and correcting errors in Quranic recitation according to **Riwayat Qaloon an Nafi** — one of the two most widely-read canonical transmissions of the Quran.

---

## What It Does

Given an audio recording of a reciter, Iqraa AI:

1. Transcribes the recitation using a fine-tuned Wav2Vec2 model
2. Compares the transcript against the canonical Qaloon text (KFGQPC edition)
3. Flags diacritical and phonetic errors specific to Qaloon rules

---

## Technical Stack

| Component | Choice |
|-----------|--------|
| Base model | `facebook/wav2vec2-large-xlsr-53-arabic` |
| Training objective | CTC (Connectionist Temporal Classification) |
| Fine-tune target | Qaloon diacritic-aware Arabic ASR |
| Training platform | Google Colab (T4 / A10G GPU) |
| Audio data | Ali al-Hudhaifi — Qaloon recitation, surah-level MP3s |
| Text reference | `qalon_canonical.json` — 6 214 ayahs, KFGQPC edition |

---

## Repository Layout

```
iqraa-ai/
├── notebooks/          # Five-stage Colab pipeline
├── src/                # Reusable Python modules
├── data/               # Text data (audio is gitignored)
├── configs/            # Training hyperparameters
└── Documentation/      # Design docs for every major step
```

---

## Quickstart

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `qalon_canonical.json` in `data/text/`
3. Place surah MP3s in `data/audio/` (gitignored — never commit)
4. Run notebooks **01 → 05** in order on Google Colab

See `Documentation/00_project_overview.md` for the full guide.

---

## Riwayat Qaloon

Qaloon is a transmission (riwaya) of the recitation of Imam Nafi al-Madani, narrated by his student Isa ibn Mina al-Qaloon (d. 835 CE). It differs from the more globally common Hafs an Asim in dozens of diacritical and phonological rules — silent letters, madd lengths, idgham patterns, and hamza handling. Correctly identifying these differences requires a model trained specifically on Qaloon-recited audio against a Qaloon-aligned reference text.
