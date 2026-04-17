# 00 — Project Overview: Iqraa AI

**Last updated:** 2026-04-17  
**Author:** Lead Developer (Claude Sonnet 4.6)  
**Status:** Active development

---

## 1. What Is Iqraa AI?

Iqraa AI is an automatic speech recognition (ASR) system designed to evaluate and correct Quranic recitation according to **Riwayat Qaloon an Nafi**. It takes a raw audio recording of a reciter and produces:

- A diacritic-aware transcript of what was actually said
- A word-level alignment against the canonical Qaloon reference text
- A structured report of phonological and diacritical errors

The end goal is a mobile/web app that gives real-time feedback to students learning to recite the Quran in the Qaloon tradition — the dominant recitation style across North Africa and parts of the Levant.

---

## 2. What Is Riwayat Qaloon and Why Does It Matter?

The Quran has been transmitted through multiple parallel chains of memorization, each preserving slightly different phonological and diacritical conventions. These chains are called **riwayat** (transmissions). The two most widely practiced worldwide are:

| Riwaya | Narrator | Primary region |
|--------|----------|----------------|
| Hafs an Asim | Hafs ibn Sulayman | Middle East, South Asia, worldwide |
| **Qaloon an Nafi** | Isa ibn Mina al-Qaloon (d. 835 CE) | **North Africa, Libya, Tunisia, parts of Levant** |

Qaloon differs from Hafs in **dozens of systematic rules**, including:

- **Hamza al-wasl** handling (silent vs. pronounced)
- **Madd** (vowel lengthening) durations — often shorter in Qaloon
- **Idgham** (assimilation) patterns unique to Nafi's chain
- Specific words where Qaloon reads a different vowel or letter entirely (the "wujuh")

A model trained on Hafs recitation (which most Arabic ASR datasets use) will produce **systematic errors** when evaluating Qaloon recitation — it will flag correct Qaloon pronunciations as mistakes. Iqraa AI solves this by training on verified Qaloon audio against a Qaloon-canonical text.

---

## 3. Technical Approach

### 3.1 Base Model

We fine-tune **`facebook/wav2vec2-large-xlsr-53-arabic`** — a 300M-parameter multilingual Wav2Vec2 model pre-trained on 53 languages including Arabic. It provides strong Arabic phoneme representations without requiring us to train from scratch.

### 3.2 Training Objective: CTC

We use **Connectionist Temporal Classification (CTC)** loss. CTC allows the model to learn alignment between variable-length audio frames and variable-length character sequences without requiring a frame-level annotation. It is the standard approach for end-to-end ASR.

### 3.3 Diacritic-Aware Vocabulary

Unlike most Arabic NLP pipelines that strip diacritics (tashkeel), we **preserve all diacritics** in the output vocabulary. This is critical for Qaloon evaluation: the difference between a fatha and a kasra on a specific word can determine whether a recitation is correct or incorrect in Qaloon's rules.

The vocabulary is built from `qalon_canonical.json` — every unique diacritized character that appears in the 6 214 ayahs becomes a token.

### 3.4 Training Platform

Training runs on **Google Colab** with a T4 or A10G GPU. The full fine-tuning pipeline is designed to complete within a single Colab session (~4–6 hours on A10G).

---

## 4. Dataset

### 4.1 Audio: Ali al-Hudhaifi (Qaloon)

- **Reciter:** Sheikh Ali ibn Abd al-Rahman al-Hudhaifi
- **Recitation:** Riwayat Qaloon an Nafi
- **Format:** Surah-level MP3 files (114 files)
- **Location:** `data/audio/` — **gitignored, never commit to GitHub**

### 4.2 Text: KFGQPC Qaloon Edition

- **File:** `qalon_canonical.json`
- **Location:** `data/text/qalon_canonical.json`
- **Content:** 6 214 ayahs, fully diacritized, King Fahd Glorious Quran Printing Complex (KFGQPC) edition for Qaloon
- **Schema:** `{ "surah": int, "ayah": int, "text": str }`

---

## 5. The Five Notebooks

The project is implemented as five sequential Colab notebooks. Run them in order.

### Notebook 01 — Data Preparation (`01_data_preparation.ipynb`)

**Goal:** Convert raw surah-level MP3s into short, labeled audio segments suitable for ASR training.

Key steps:
- Load MP3s with `librosa`
- Normalize loudness with `pyloudnorm` (target: -23 LUFS)
- Convert to 16 kHz mono WAV (Wav2Vec2 requirement)
- Split into ayah-level segments using forced alignment or energy-based segmentation
- Output: `data/processed/segments/` — one WAV per ayah

### Notebook 02 — Text Processing (`02_text_processing.ipynb`)

**Goal:** Prepare the canonical text for use as CTC training labels.

Key steps:
- Load `qalon_canonical.json`
- Apply Qaloon-specific normalization rules (via `src/normalize_text.py`)
- Build the character-level vocabulary (via `src/build_vocab.py`)
- Output: `data/processed/labels.json`, `data/processed/vocab.json`

### Notebook 03 — Alignment (`03_alignment.ipynb`)

**Goal:** Pair each audio segment with its correct text label and build the HuggingFace `Dataset`.

Key steps:
- Match segment filenames to ayah IDs
- Validate that audio length is plausible for the text length
- Build a `datasets.Dataset` with `audio` and `text` columns
- Push to HuggingFace Hub (optional) or save locally
- Output: `data/processed/hf_dataset/`

### Notebook 04 — Training (`04_training.ipynb`)

**Goal:** Fine-tune Wav2Vec2 on the aligned Qaloon dataset.

Key steps:
- Load base model + processor from HuggingFace
- Freeze feature extractor layers (standard practice)
- Configure CTC training with `Wav2Vec2ForCTC`
- Train with `Trainer` API using `configs/training_config.yaml`
- Save checkpoint to `models/qaloon_wav2vec2/`

### Notebook 05 — Evaluation (`05_evaluation.ipynb`)

**Goal:** Measure model accuracy and diagnose error patterns.

Key steps:
- Load the saved checkpoint
- Run inference on a held-out test set
- Compute **WER** (Word Error Rate) and **CER** (Character Error Rate) with `jiwer`
- Analyze errors specific to Qaloon rules (madd, hamza, idgham)
- Output: evaluation report in `Documentation/`

---

## 6. How to Run the Project End-to-End

### Prerequisites

- A Google account with access to Google Colab
- The surah MP3 files (Hudhaifi, Qaloon)
- `qalon_canonical.json`

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/iqraa-ai.git
cd iqraa-ai

# 2. Upload to Google Drive (for Colab access)
#    Place the entire repo folder in My Drive/iqraa-ai/
#    Place audio files in My Drive/iqraa-ai/data/audio/

# 3. Open each notebook in Colab in order:
#    notebooks/01_data_preparation.ipynb
#    notebooks/02_text_processing.ipynb
#    notebooks/03_alignment.ipynb
#    notebooks/04_training.ipynb
#    notebooks/05_evaluation.ipynb

# 4. Each notebook installs its own dependencies via:
#    !pip install -r /content/iqraa-ai/requirements.txt

# 5. Results land in data/processed/ and models/
```

### Local Development (Windows 11, no GPU)

For editing source modules and running unit tests locally:

```bash
pip install -r requirements.txt
python -m pytest src/tests/  # once tests are written
```

Heavy computation (training, audio processing) must run on Colab.

---

## 7. File-by-File Reference

| File | Purpose |
|------|---------|
| `src/normalize_text.py` | Qaloon-specific Arabic text normalization |
| `src/build_vocab.py` | Build CTC character vocabulary from text corpus |
| `src/dataset.py` | HuggingFace Dataset construction and loading |
| `src/train.py` | Training loop (callable from notebook or CLI) |
| `src/evaluate.py` | WER/CER computation and error analysis |
| `configs/training_config.yaml` | All hyperparameters in one place |
| `data/text/qalon_canonical.json` | Canonical Qaloon reference text (6 214 ayahs) |
| `data/audio/` | Audio files — **never commit** |
