"""
Fine-tuning entry point for Iqraa AI.

Can be called directly from Notebook 04 or run as a standalone script on
a GPU machine.  All hyperparameters are read from configs/training_config.yaml
so this file rarely needs to change.

Architecture: Wav2Vec2ForCTC with a diacritic-aware Arabic vocabulary.
Training objective: CTC loss.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml
from datasets import DatasetDict
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_processor(vocab_path: str, model_id: str) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_values = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def preprocess_dataset(ds: DatasetDict, processor: Wav2Vec2Processor) -> DatasetDict:
    def prepare(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    return ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=1)


def train(config_path: str, vocab_path: str, dataset_dir: str) -> None:
    from src.dataset import load_dataset_from_disk

    cfg = load_config(config_path)
    ds = load_dataset_from_disk(dataset_dir)
    processor = build_processor(vocab_path, cfg["model"]["base_model_id"])

    ds = preprocess_dataset(ds, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        cfg["model"]["base_model_id"],
        ctc_loss_reduction=cfg["ctc"]["ctc_loss_reduction"],
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    if cfg["model"]["freeze_feature_extractor"]:
        model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        fp16=cfg["training"]["fp16"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        greater_is_better=cfg["training"]["greater_is_better"],
    )

    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    model.save_pretrained(cfg["training"]["output_dir"])
    processor.save_pretrained(cfg["training"]["output_dir"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--vocab", default="data/processed/vocab.json")
    parser.add_argument("--dataset", default="data/processed/hf_dataset")
    args = parser.parse_args()
    train(args.config, args.vocab, args.dataset)
