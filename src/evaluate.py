"""
Evaluation module for Iqraa AI.

Computes Word Error Rate (WER) and Character Error Rate (CER) on a test set
and generates an error analysis report focused on Qaloon-specific patterns.

WER and CER are computed with the `jiwer` library.  The error analysis
breaks down mistakes by error type: substitutions, deletions, insertions.
"""

import json
from pathlib import Path

import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


_TRANSFORMS = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def load_model_and_processor(model_dir: str):
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    model.eval()
    return model, processor


def transcribe_batch(
    audio_arrays: list,
    sampling_rates: list[int],
    model,
    processor: Wav2Vec2Processor,
) -> list[str]:
    import torch

    inputs = processor(
        audio_arrays,
        sampling_rate=sampling_rates[0],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict:
    wer = jiwer.wer(references, hypotheses, reference_transform=_TRANSFORMS,
                    hypothesis_transform=_TRANSFORMS)
    cer = jiwer.cer(references, hypotheses, reference_transform=_TRANSFORMS,
                    hypothesis_transform=_TRANSFORMS)
    return {"wer": round(wer, 4), "cer": round(cer, 4)}


def evaluate_dataset(model_dir: str, dataset_dir: str, output_path: str | None = None) -> dict:
    """
    Run full evaluation on the test split and return metrics dict.

    Parameters
    ----------
    model_dir : str
        Path to saved model checkpoint (output of training).
    dataset_dir : str
        Path to HuggingFace DatasetDict saved to disk.
    output_path : str | None
        If given, write JSON report to this path.
    """
    from src.dataset import load_dataset_from_disk

    model, processor = load_model_and_processor(model_dir)
    ds = load_dataset_from_disk(dataset_dir)
    test = ds["test"]

    references, hypotheses = [], []
    for sample in test:
        audio = sample["audio"]
        hyp = transcribe_batch([audio["array"]], [audio["sampling_rate"]], model, processor)[0]
        references.append(sample["text"])
        hypotheses.append(hyp)

    metrics = compute_metrics(references, hypotheses)
    metrics["num_samples"] = len(references)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qaloon_wav2vec2")
    parser.add_argument("--dataset", default="data/processed/hf_dataset")
    parser.add_argument("--output", default="Documentation/evaluation_report.json")
    args = parser.parse_args()

    metrics = evaluate_dataset(args.model, args.dataset, args.output)
    print(f"WER: {metrics['wer']:.2%}  |  CER: {metrics['cer']:.2%}")
