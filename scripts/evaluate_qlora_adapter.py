#!/usr/bin/env python3
"""Generate and score validation outputs from a saved MURL QLoRA adapter."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_sft_dataset import user_message
from scripts.prepare_run import prepare_run


def load_config(path: Path):
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_system_prompt(root: Path, relative_path: str) -> str:
    prompt = (root / relative_path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"SFT system prompt is empty: {relative_path}")
    return prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora-v0.1-smoke.yaml"))
    parser.add_argument("--adapter", type=Path, help="Saved adapter directory; omit for the untouched base-model control.")
    parser.add_argument("--run-id", help="Override the configured validation run ID.")
    args = parser.parse_args()
    config = load_config(args.config)
    inference = config["adapter_inference"]
    if inference["split"] != "validation":
        raise ValueError("Adapter evaluation is restricted to the validation split.")
    system_prompt = load_system_prompt(ROOT, inference["system_prompt_file"])

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU.")
    base = config["base_model"]
    compute_dtype = torch.bfloat16 if base["compute_dtype"] == "bfloat16" else torch.float16
    set_seed(config["run"]["seed"])
    run_id = args.run_id or inference["run_id"]
    examples, _, run_dir = prepare_run(
        ROOT, run_id, inference["split"], inference["system_prompt_file"], None,
        metadata={
            "model_id": base["id"], "model_revision": base["revision"], "adapter": str(args.adapter) if args.adapter else None,
            "seed": config["run"]["seed"], "gpu": torch.cuda.get_device_name(0),
            "packages": {name: version(name) for name in ["torch", "transformers", "accelerate", "bitsandbytes", "peft"]},
            "quantization": {"load_in_4bit": True, "quant_type": base["quantization"], "compute_dtype": str(compute_dtype)},
            "decoding": {"do_sample": False, "max_new_tokens": inference["max_new_tokens"]},
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base["id"], revision=base["revision"], token=False,
        trust_remote_code=base.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=base["quantization"], bnb_4bit_compute_dtype=compute_dtype)
    model_loader = base.get("model_loader", "causal_lm")
    model_class = {
        "causal_lm": AutoModelForCausalLM,
        "image_text_to_text": AutoModelForImageTextToText,
    }.get(model_loader)
    if model_class is None:
        raise ValueError(f"Unsupported base_model.model_loader: {model_loader}")
    model = model_class.from_pretrained(
        base["id"], revision=base["revision"], token=False, device_map="auto",
        quantization_config=quantization, torch_dtype=compute_dtype,
        trust_remote_code=base.get("trust_remote_code", False),
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    outputs_path = run_dir / "outputs.jsonl"
    with outputs_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, 1):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message(example)}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=inference["max_new_tokens"], do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            output = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            handle.write(json.dumps({"example_id": example["example_id"], "output": output}, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(examples):
                print(f"Completed {index}/{len(examples)}")

    results_dir = ROOT / "results" / run_id
    subprocess.run([
        sys.executable, "evaluation/evaluate.py", "--gold", str(run_dir / "gold.jsonl"),
        "--outputs", str(outputs_path), "--run-id", run_id, "--results-dir", str(ROOT / "results"),
    ], check=True, cwd=ROOT)
    shutil.copy2(run_dir / "metadata.json", results_dir / "metadata.json")
    shutil.copytree(results_dir, run_dir / "results")
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=ROOT, base_dir=str(run_dir.relative_to(ROOT)))
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
