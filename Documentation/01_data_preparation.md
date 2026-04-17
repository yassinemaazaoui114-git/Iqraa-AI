# 01 — Data Preparation

**Notebook:** `notebooks/01_data_preparation.ipynb`  
**Last updated:** 2026-04-17  
**Status:** Complete

---

## Purpose

Notebook 01 is the entry point of the Iqraa AI training pipeline. It converts
raw surah-level MP3 recordings into properly formatted audio files and builds a
structured manifest linking every audio file to its canonical Qaloon text.

Nothing in this notebook trains or modifies the model. It is pure data I/O and
format conversion.

---

## Inputs

| Input | Location on Google Drive | Description |
|-------|--------------------------|-------------|
| Surah MP3 files | `IqraaAI_Dataset/audio/hudhaifi_qaloon/` | 114 files named `001.mp3` – `114.mp3`, Ali al-Hudhaifi, Riwayat Qaloon |
| Canonical text | `IqraaAI_Dataset/text/qalon_canonical.json` | 6 214-entry dict, keys `SSSAAA` → diacritized Arabic string |

### qalon_canonical.json schema

```
{
  "001001": "اِ۬لْحَمْدُ لِلهِ رَبِّ اِ۬لْعَٰلَمِينَ ...",
  "001002": "...",
  ...
  "114006": "..."
}
```

Keys are 6-character strings: first 3 digits = surah number (zero-padded),
last 3 digits = ayah number (zero-padded). Values are fully diacritized Arabic
strings including the KFGQPC ayah numeral at the end (stripped in Notebook 02).

---

## Outputs

All outputs are written to `IqraaAI_Dataset/processed/` on Google Drive.

| Output | Path | Description |
|--------|------|-------------|
| WAV files | `processed/wav/001.wav` – `114.wav` | 16 kHz mono PCM-16, −23 LUFS normalised |
| Pairing manifest | `processed/pairing_manifest.json` | Structured list linking each WAV to its ayahs |
| Completion marker | `processed/notebook_01_complete.json` | Summary with counts for downstream validation |

### pairing_manifest.json schema

```json
[
  {
    "surah": 1,
    "audio_path": "/content/drive/.../processed/wav/001.wav",
    "ayahs": [
      { "key": "001001", "ayah_num": 1, "text": "..." },
      { "key": "001002", "ayah_num": 2, "text": "..." },
      ...
    ]
  },
  ...
]
```

---

## What Each Cell Does

| Cell | Type | What it does |
|------|------|--------------|
| 1 | Markdown | Title, purpose, and re-run safety note |
| 2 | Code | Mount Google Drive; define all path constants |
| 3 | Markdown | Explains the four libraries being installed and why |
| 4 | Code | `pip install librosa soundfile pyloudnorm tqdm` |
| 5 | Markdown | Why MP3 is problematic for Wav2Vec2; the conversion plan |
| 6 | Code | Count MP3s, print first 5 filenames, total size, sample duration |
| 7 | Markdown | What 16 kHz mono means; what LUFS normalisation is |
| 8 | Code | Convert all MP3s → 16 kHz mono WAV at −23 LUFS; skip already-done files |
| 9 | Markdown | qalon_canonical.json schema and how it pairs with audio |
| 10 | Code | Load JSON; show first 3 entries; ayah counts for surahs 1–5; assert Al-Fatihah = 7 |
| 11 | Markdown | Pairing logic and manifest structure explanation |
| 12 | Code | Build `pairing_manifest.json`; report missing audio/text |
| 13 | Markdown | Summary of outputs and preview of what Notebook 02 does |
| 14 | Code | Write `notebook_01_complete.json` completion marker |

---

## Key Technical Decisions

### 16 kHz mono WAV
`facebook/wav2vec2-large-xlsr-53-arabic` was pre-trained on 16 kHz mono audio.
Providing audio at any other sample rate causes silent accuracy degradation
because the CNN feature extractor operates on a fixed number of frames per
second. We resample during the `librosa.load` call (not as a separate step) to
keep the pipeline simple.

### −23 LUFS normalisation
EBU R128 broadcast standard. Different surah recordings vary in loudness; without
normalisation the model would see inconsistent input amplitude across the corpus.
`pyloudnorm` measures integrated loudness (perceptual, time-averaged) and scales
the waveform to match the target. A final `np.clip(audio, -1.0, 1.0)` prevents
overshoot on very short or very quiet files.

### PCM_16 WAV subtype
We write 16-bit integer PCM (the standard WAV format). This is lossless within
the 16-bit range and produces files roughly 30 × smaller than 32-bit float WAV
while being indistinguishable for speech-band audio.

### Skip-if-exists logic
Cell 8 checks for the output WAV before converting. This makes the notebook safe
to re-run after partial failures without re-processing completed files.

### Manifest instead of direct segmentation
This notebook produces a surah-level manifest rather than ayah-level segments.
Ayah segmentation (finding where each ayah starts and ends within the surah WAV)
is the job of Notebook 03 (alignment), which can use more sophisticated methods.
Separating the two steps means we can re-run alignment without re-running the
slow MP3→WAV conversion.

---

## How to Run

### Prerequisites

1. A Google account with Google Colab access
2. A Google Drive folder `My Drive/IqraaAI_Dataset/` with:
   - `audio/hudhaifi_qaloon/001.mp3` – `114.mp3`
   - `text/qalon_canonical.json`

### Steps

1. Open [Google Colab](https://colab.research.google.com/)
2. File → Open notebook → GitHub → paste the repo URL → select
   `notebooks/01_data_preparation.ipynb`
3. Runtime → Change runtime type → T4 GPU (not required for this notebook, but
   keeps the session consistent with later notebooks)
4. Run All Cells (Runtime → Run all), or run cells one by one in order
5. When Cell 2 runs, a Drive auth prompt will appear — approve it
6. Cell 8 will take 20–60 minutes depending on audio file sizes

### Expected output

```
Converted      : 114 files
Total WAV size : ~800 MB
Surahs paired  : 114
Surahs missing audio : 0  ✓
Surahs missing text  : 0  ✓
Notebook 01 complete.
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `No MP3 files found` | Wrong path in `AUDIO_DIR` | Check Drive folder name matches `DRIVE_BASE` |
| `pyloudnorm` error on short file | File < 0.5 s or silence-only | Check the MP3 is not corrupt; these files are skipped with a printed ERROR |
| Cell 4 install hangs | Colab network issue | Interrupt and re-run |
| Drive auth loop | Third-party cookies blocked | Allow cookies for `colab.research.google.com` |

---

## Next Step

Run **Notebook 02 — Text Processing** (`notebooks/02_text_processing.ipynb`).
