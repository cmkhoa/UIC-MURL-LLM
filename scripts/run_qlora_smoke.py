#!/usr/bin/env python3
"""Run the configured MURL QLoRA smoke training and verify the saved adapter loads."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML object")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora-v0.1-smoke.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA smoke training requires a CUDA GPU.")

    run = config["run"]
    base = config["base_model"]
    data = config["data"]
    qlora = config["qlora"]
    training = config["training"]
    train_file = Path(data["train_file"])
    output_dir = Path(training["output_dir"])
    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if sha256(train_file) != data["train_file_sha256"]:
        raise ValueError("Training dataset hash does not match the frozen configuration.")

    records = read_jsonl(train_file)
    if len(records) != data["train_examples"]:
        raise ValueError("Training example count does not match the frozen configuration.")
    random.seed(run["seed"])
    set_seed(run["seed"])

    compute_dtype = torch.bfloat16 if base["compute_dtype"] == "bfloat16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=base["load_in_4bit"],
        bnb_4bit_quant_type=base["quantization"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(base["id"], revision=base["revision"], token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base["id"], revision=base["revision"], token=False, device_map="auto",
        quantization_config=quantization, torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=qlora["rank"], lora_alpha=qlora["alpha"], lora_dropout=qlora["dropout"],
        target_modules=qlora["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))

    class ChatDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            messages = self.rows[index][data["message_field"]]
            prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = tokenizer(
                full, add_special_tokens=False, truncation=True, max_length=training["max_sequence_length"],
            )["input_ids"]
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True, max_length=training["max_sequence_length"],
            )["input_ids"]
            prompt_length = min(len(prompt_ids), len(input_ids))
            if prompt_length >= len(input_ids):
                raise ValueError(f"{self.rows[index]['example_id']}: assistant target was truncated")
            return {"input_ids": input_ids, "labels": [-100] * prompt_length + input_ids[prompt_length:]}

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir), max_steps=training["max_steps"], learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"], warmup_steps=training["warmup_steps"],
        logging_steps=training["logging_steps"], save_steps=training["save_steps"], save_total_limit=1,
        report_to="none", fp16=compute_dtype == torch.float16, bf16=compute_dtype == torch.bfloat16,
        remove_unused_columns=False, seed=run["seed"], data_seed=run["seed"],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ChatDataset(records), data_collator=collate)
    result = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))}
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    verify_base = AutoModelForCausalLM.from_pretrained(
        base["id"], revision=base["revision"], token=False, device_map="auto",
        quantization_config=quantization, torch_dtype=compute_dtype,
    )
    verified_adapter = PeftModel.from_pretrained(verify_base, adapter_dir)
    metadata = {
        "run_id": run["id"], "dataset_sha256": sha256(train_file), "examples": len(records),
        "base_model": base["id"], "base_revision": base["revision"], "seed": run["seed"],
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__,
        "adapter_load_verified": isinstance(verified_adapter, PeftModel), "training_metrics": metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
