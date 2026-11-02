"""Run one v0.8 paired cross-family experiment in a mounted Colab session.

Set MURL_V08_FAMILY and MURL_V08_V05_CONFIG before execution. Optionally set
MURL_V08_V07_CONFIG to add the independent v0.7 arm. The script always
validates one untouched base before the requested adapter arm(s).
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def run_logged(args):
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, args)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


family = os.environ["MURL_V08_FAMILY"]
run_label = os.environ.get("MURL_V08_RUN_LABEL", family)
v05_config = os.environ["MURL_V08_V05_CONFIG"]
v07_config = os.environ.get("MURL_V08_V07_CONFIG")
drive = Path("/content/drive/MyDrive/MURL")
assert drive.parent.is_dir(), "Mount Drive before running"
os.environ["HF_HOME"] = str(drive / "huggingface-cache")
archive = Path("/content/murl-v08-source.zip")
root = Path(f"/content/UIC-MURL-v08-{run_label}")
assert not root.exists(), "Use a fresh source directory"
with zipfile.ZipFile(archive) as source:
    assert source.testzip() is None
    source.extractall(root)
backup = drive / "runs/v0.8-cross-family" / run_label
backup.mkdir(parents=True, exist_ok=True)
source_backup = backup / archive.name
shutil.copy2(archive, source_backup)
assert file_sha256(archive) == file_sha256(source_backup)
print("source", file_sha256(archive), flush=True)
os.chdir(root)
run_logged([
    sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.1",
    "accelerate==1.15.0", "bitsandbytes==0.50.2", "peft==0.21.2", "pyyaml",
])

from huggingface_hub import snapshot_download
import yaml

v05 = yaml.safe_load(Path(v05_config).read_text())
v07 = yaml.safe_load(Path(v07_config).read_text()) if v07_config else None
if v07:
    for key in ("id", "revision", "quantization", "compute_dtype", "trust_remote_code"):
        assert v05["base_model"].get(key) == v07["base_model"].get(key), key
    for key in ("train_file", "train_file_sha256", "validation_file", "validation_file_sha256"):
        assert v05["data"][key] == v07["data"][key], key
    assert v05["qlora"] == v07["qlora"]

base = v05["base_model"]
snapshot = Path(snapshot_download(base["id"], revision=base["revision"], token=False))
local_cache = Path(f"/content/murl-hf-v08-{family}")
local_snapshot = local_cache / ("models--" + base["id"].replace("/", "--")) / "snapshots" / base["revision"]
local_snapshot.mkdir(parents=True, exist_ok=True)
for file in snapshot.iterdir():
    source = file
    for _ in range(8):
        if source.is_symlink():
            source = source.parent / os.readlink(source)
        elif source.is_file() and source.stat().st_size < 512 and source.read_bytes().startswith(b"../../blobs/"):
            source = source.parent / source.read_text().strip()
        else:
            break
    else:
        raise ValueError(f"Unresolved cache link: {file}")
    shutil.copyfile(source, local_snapshot / file.name)
from safetensors import safe_open
for file in local_snapshot.glob("*.safetensors"):
    with safe_open(file, framework="pt", device="cpu") as weights:
        print(file.name, file.stat().st_size, len(weights.keys()), "tensors verified", flush=True)
os.environ["HF_HUB_CACHE"] = str(local_cache)
os.environ["HF_HUB_OFFLINE"] = "1"

assert file_sha256(v05["data"]["validation_file"]) == v05["data"]["validation_file_sha256"]


def preserve(source, name):
    destination = Path("/content") / name
    shutil.copy2(source, destination)
    with zipfile.ZipFile(destination) as saved:
        assert saved.testzip() is None
    drive_destination = backup / name
    shutil.copy2(destination, drive_destination)
    local_hash = file_sha256(destination)
    drive_hash = file_sha256(drive_destination)
    assert local_hash == drive_hash
    print(name, local_hash, "local/Drive match", flush=True)


base_run = f"{family}-murl-v0.8-base-control-validation"
run_logged([
    sys.executable, "-u", "scripts/evaluate_qlora_adapter.py", "--config", v05_config,
    "--run-id", base_run,
])
preserve(root / "runs" / f"{base_run}.zip", f"{family}-v08-base-control-validation-results.zip")


def run_arm(label, config, settings):
    run_logged([sys.executable, "-u", "scripts/run_qlora_smoke.py", "--config", config])
    model_dir = Path(settings["training"]["output_dir"])
    shutil.copy2(archive, model_dir / "source.zip")
    shutil.copy2("data/training/v0.5-sft-train.metadata.json", model_dir / "dataset_metadata.json")
    training_zip = shutil.make_archive(f"/content/{family}-v08-{label}-training", "zip", root_dir=root, base_dir=model_dir)
    preserve(training_zip, f"{family}-v08-{label}-training-results.zip")
    shutil.copytree(model_dir / "adapter", backup / f"{label}-adapter", dirs_exist_ok=True)
    if label == "v07":
        shutil.copytree(model_dir / "checkpoint-30", backup / "v07-stage-1-checkpoint", dirs_exist_ok=True)
    run_logged([
        sys.executable, "-u", "scripts/evaluate_qlora_adapter.py", "--config", config,
        "--adapter", str(model_dir / "adapter"),
    ])
    preserve(
        root / "runs" / (settings["adapter_inference"]["run_id"] + ".zip"),
        f"{family}-v08-{label}-validation-results.zip",
    )


run_arm("v05", v05_config, v05)
if v07:
    run_arm("v07", v07_config, v07)
print("EXPERIMENT COMPLETE: requested archives verified and saved to Drive.", flush=True)
