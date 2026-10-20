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

from scripts.make_sft_dataset import SYSTEM_PROMPT, user_message
from scripts.prepare_run import prepare_run


def load_config(path: Path):
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora-v0.1-smoke.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    inference = config["adapter_inference"]
    if inference["split"] != "validation":
        raise ValueError("Adapter evaluation is restricted to the validation split.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU.")
    base = config["base_model"]
    compute_dtype = torch.bfloat16 if base["compute_dtype"] == "bfloat16" else torch.float16
    set_seed(config["run"]["seed"])
    examples, _, run_dir = prepare_run(
        ROOT, inference["run_id"], inference["split"], inference["system_prompt_file"], None,
        metadata={
            "model_id": base["id"], "model_revision": base["revision"], "adapter": str(args.adapter),
            "seed": config["run"]["seed"], "gpu": torch.cuda.get_device_name(0),
            "packages": {name: version(name) for name in ["torch", "transformers", "accelerate", "bitsandbytes", "peft"]},
            "quantization": {"load_in_4bit": True, "quant_type": base["quantization"], "compute_dtype": str(compute_dtype)},
            "decoding": {"do_sample": False, "max_new_tokens": inference["max_new_tokens"]},
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(base["id"], revision=base["revision"], token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=base["quantization"], bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base["id"], revision=base["revision"], token=False, device_map="auto",
        quantization_config=quantization, torch_dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    outputs_path = run_dir / "outputs.jsonl"
    with outputs_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, 1):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message(example)}]
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

    results_dir = ROOT / "results" / inference["run_id"]
    subprocess.run([
        sys.executable, "evaluation/evaluate.py", "--gold", str(run_dir / "gold.jsonl"),
        "--outputs", str(outputs_path), "--run-id", inference["run_id"], "--results-dir", str(ROOT / "results"),
    ], check=True, cwd=ROOT)
    shutil.copy2(run_dir / "metadata.json", results_dir / "metadata.json")
    shutil.copytree(results_dir, run_dir / "results")
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=ROOT, base_dir=str(run_dir.relative_to(ROOT)))
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
