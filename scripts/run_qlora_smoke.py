#!/usr/bin/env python3
"""Run the configured MURL QLoRA smoke training and verify the saved adapter loads."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
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


def resolve_weighted_schedule(training: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize the supported two-phase loss schedules."""
    hierarchical = training.get("hierarchical_multitask")
    two_stage = training.get("two_stage_diagnosis")
    if hierarchical and two_stage:
        raise ValueError("Configure only one weighted training schedule.")
    if hierarchical:
        return {
            "first_steps": int(hierarchical["decision_phase"]["steps"]),
            "first_weights": hierarchical["decision_phase"]["field_weights"],
            "second_weights": hierarchical["diagnosis_phase"]["field_weights"],
            "reload_at_boundary": False,
        }
    if two_stage:
        first_steps = int(two_stage["stage_1"]["steps"])
        second_steps = int(two_stage["stage_2"]["steps"])
        if first_steps + second_steps != int(training["max_steps"]):
            raise ValueError("Two-stage optimizer steps must sum to training.max_steps.")
        if int(training["save_steps"]) != first_steps:
            raise ValueError("Two-stage save_steps must equal the Stage 1 boundary.")
        return {
            "first_steps": first_steps,
            "first_weights": two_stage["stage_1"]["field_weights"],
            "second_weights": two_stage["stage_2"]["field_weights"],
            "reload_at_boundary": True,
        }
    return None


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora-v0.1-smoke.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
        TrainerCallback, TrainingArguments, set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA smoke training requires a CUDA GPU.")

    run = config["run"]
    base = config["base_model"]
    data = config["data"]
    qlora = config["qlora"]
    training = config["training"]
    weighted_schedule = resolve_weighted_schedule(training)
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
    def build_model():
        loaded = AutoModelForCausalLM.from_pretrained(
            base["id"], revision=base["revision"], token=False, device_map="auto",
            quantization_config=quantization, torch_dtype=compute_dtype,
        )
        loaded.config.use_cache = False
        loaded = prepare_model_for_kbit_training(loaded)
        return get_peft_model(loaded, LoraConfig(
            r=qlora["rank"], lora_alpha=qlora["alpha"], lora_dropout=qlora["dropout"],
            target_modules=qlora["target_modules"], bias="none", task_type="CAUSAL_LM",
        ))

    model = build_model()

    class ChatDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            messages = self.rows[index][data["message_field"]]
            prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoded = tokenizer(full, add_special_tokens=False, truncation=True,
                                max_length=training["max_sequence_length"],
                                return_offsets_mapping=bool(weighted_schedule))
            input_ids = encoded["input_ids"]
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True, max_length=training["max_sequence_length"],
            )["input_ids"]
            prompt_length = min(len(prompt_ids), len(input_ids))
            if prompt_length >= len(input_ids):
                raise ValueError(f"{self.rows[index]['example_id']}: assistant target was truncated")
            result = {"input_ids": input_ids, "labels": [-100] * prompt_length + input_ids[prompt_length:]}
            if weighted_schedule:
                target = messages[-1]["content"]
                start = full.rfind(target)
                if start < 0:
                    raise ValueError(f"{self.rows[index]['example_id']}: cannot locate assistant target")
                spans = {}
                for field in ("overall_correctness", "first_error_step", "error_taxonomy"):
                    match = re.search(rf'"{field}":[^,}}]+', target)
                    if not match:
                        raise ValueError(f"{self.rows[index]['example_id']}: missing target field {field}")
                    spans[field] = (start + match.start(), start + match.end())
                def weights(field_weights: dict[str, float]) -> list[float]:
                    result_weights = [0.0] * len(input_ids)
                    for token_index, (left, right) in enumerate(encoded["offset_mapping"]):
                        if token_index < prompt_length or left == right:
                            continue
                        weight = 1.0
                        for field, (field_left, field_right) in spans.items():
                            if left < field_right and right > field_left:
                                weight = float(field_weights[field])
                        result_weights[token_index] = weight
                    return result_weights
                result["first_phase_weights"] = weights(weighted_schedule["first_weights"])
                result["second_phase_weights"] = weights(weighted_schedule["second_weights"])
            return result

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        first_phase_weights, second_phase_weights = [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            if weighted_schedule:
                first_phase_weights.append(feature["first_phase_weights"] + [0.0] * padding)
                second_phase_weights.append(feature["second_phase_weights"] + [0.0] * padding)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if weighted_schedule:
            batch["first_phase_weights"] = torch.tensor(first_phase_weights, dtype=torch.float)
            batch["second_phase_weights"] = torch.tensor(second_phase_weights, dtype=torch.float)
        return batch

    class WeightedStageTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            first_phase_weights = inputs.pop("first_phase_weights")
            second_phase_weights = inputs.pop("second_phase_weights")
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
            weights = (first_phase_weights if self.state.global_step < weighted_schedule["first_steps"]
                       else second_phase_weights)[:, 1:].contiguous()
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none", ignore_index=-100,
            ).view_as(targets)
            active = (targets != -100).to(weights.dtype) * weights
            loss = (losses * active).sum() / active.sum().clamp_min(1.0)
            return (loss, outputs) if return_outputs else loss

    class StageBoundaryCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == weighted_schedule["first_steps"]:
                control.should_save = True
                control.should_training_stop = True
            return control

    training_args = TrainingArguments(
        output_dir=str(output_dir), max_steps=training["max_steps"], learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"], warmup_steps=training["warmup_steps"],
        logging_steps=training["logging_steps"], save_steps=training["save_steps"],
        save_total_limit=training.get("save_total_limit", 1),
        report_to="none", fp16=compute_dtype == torch.float16, bf16=compute_dtype == torch.bfloat16,
        remove_unused_columns=False, seed=run["seed"], data_seed=run["seed"],
    )
    trainer_class = WeightedStageTrainer if weighted_schedule else Trainer
    dataset = ChatDataset(records)
    stage_metrics = {}
    callbacks = [StageBoundaryCallback()] if weighted_schedule and weighted_schedule["reload_at_boundary"] else []
    trainer = trainer_class(
        model=model, args=training_args, train_dataset=dataset, data_collator=collate,
        callbacks=callbacks,
    )
    result = trainer.train()
    stage_checkpoint = None
    if weighted_schedule and weighted_schedule["reload_at_boundary"]:
        if trainer.state.global_step != weighted_schedule["first_steps"]:
            raise RuntimeError("Stage 1 did not stop at its precommitted boundary.")
        stage_checkpoint = output_dir / f"checkpoint-{weighted_schedule['first_steps']}"
        if not stage_checkpoint.is_dir():
            raise FileNotFoundError(f"Missing Stage 1 checkpoint: {stage_checkpoint}")
        stage_metrics["stage_1"] = {
            key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))
        }
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        model = build_model()
        trainer = trainer_class(
            model=model, args=training_args, train_dataset=dataset, data_collator=collate,
        )
        result = trainer.train(resume_from_checkpoint=str(stage_checkpoint))
        if trainer.state.global_step != training["max_steps"]:
            raise RuntimeError("Stage 2 did not reach the total optimizer-step budget.")
        stage_metrics["stage_2"] = {
            key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))
        }
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))}
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if stage_metrics:
        (output_dir / "stage_metrics.json").write_text(
            json.dumps(stage_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

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
    if stage_checkpoint:
        metadata["stage_1_checkpoint"] = str(stage_checkpoint)
        metadata["stage_1_checkpoint_sha256"] = tree_sha256(stage_checkpoint)
        metadata["stage_2_adapter_sha256"] = tree_sha256(adapter_dir)
        metadata["optimizer_scheduler_resumed"] = True
        metadata["stage_metrics"] = stage_metrics
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
