"""
HuggingFace Dataset construction for Iqraa AI.

This module pairs each processed audio segment (16 kHz WAV) with its
corresponding diacritized Qaloon text label, then wraps everything in
a datasets.DatasetDict with train/test splits.

The resulting dataset is the direct input to Notebook 04 (training).
"""

import json
from pathlib import Path

import datasets
from datasets import Audio, Dataset, DatasetDict


def load_labels(labels_path: str) -> dict[str, str]:
    """
    Load the ayah-ID → normalized-text mapping produced by Notebook 02.

    Returns
    -------
    dict
        Keys are segment IDs like "001_001" (surah_ayah), values are
        diacritized Arabic strings.
    """
    with open(labels_path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(
    segments_dir: str,
    labels_path: str,
    test_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Build a HuggingFace DatasetDict from audio segments and text labels.

    Parameters
    ----------
    segments_dir : str
        Directory containing WAV files named "<surah>_<ayah>.wav".
    labels_path : str
        Path to labels.json produced by src/build_vocab or Notebook 02.
    test_split : float
        Fraction of data held out for evaluation.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    DatasetDict with "train" and "test" splits, each having columns:
        - audio : datasets.Audio (16 kHz)
        - text  : str (diacritized Qaloon label)
        - id    : str (segment identifier)
    """
    labels = load_labels(labels_path)
    segments_dir = Path(segments_dir)

    records = []
    for wav_path in sorted(segments_dir.glob("*.wav")):
        segment_id = wav_path.stem  # e.g. "001_001"
        if segment_id not in labels:
            continue
        records.append({
            "id": segment_id,
            "audio": str(wav_path),
            "text": labels[segment_id],
        })

    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    split = ds.train_test_split(test_size=test_split, seed=seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def save_dataset(ds: DatasetDict, output_dir: str) -> None:
    ds.save_to_disk(output_dir)


def load_dataset_from_disk(dataset_dir: str) -> DatasetDict:
    return datasets.load_from_disk(dataset_dir)
