"""Execute the frozen v0.4 experiment after uploading its source ZIP and mounting Drive."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def run_logged(args):
    """Stream subprocess output into the executed notebook, including failures."""
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, args)

drive = Path('/content/drive/MyDrive/MURL')
assert drive.parent.is_dir(), 'Mount Drive before running'
drive.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(drive / 'huggingface-cache')
archive = Path('/content/murl-v04-source.zip')
root = Path('/content/UIC-MURL-LLM')
assert not root.exists(), 'Use a fresh source directory'
with zipfile.ZipFile(archive) as source:
    assert source.testzip() is None
    source.extractall(root)
backup = drive / 'runs/v0.4-taxonomy-balanced'
backup.mkdir(parents=True, exist_ok=True)
shutil.copy2(archive, backup / archive.name)
os.chdir(root)
run_logged([
    sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==5.20.1', 'accelerate==1.15.0', 'bitsandbytes==0.50.2',
    'peft==0.21.2', 'pyyaml',
])

# Download once into Drive; all subsequent loads and sessions reuse this cache.
from huggingface_hub import snapshot_download
revision = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
snapshot = Path(snapshot_download('Qwen/Qwen2.5-3B-Instruct', revision=revision, token=False))
# Drive can expose nested cache links as tiny text files. Materialize actual
# blobs locally while keeping HF_HOME and the original weights persistent.
local_cache = Path('/content/murl-hf-hub')
local_snapshot = local_cache / 'models--Qwen--Qwen2.5-3B-Instruct/snapshots' / revision
local_snapshot.mkdir(parents=True, exist_ok=True)
for file in snapshot.iterdir():
    source = file
    for _ in range(8):
        if source.is_symlink():
            source = source.parent / os.readlink(source)
        elif source.is_file() and source.stat().st_size < 512 and source.read_bytes().startswith(b'../../blobs/'):
            source = source.parent / source.read_text().strip()
        else:
            break
    else:
        raise ValueError(f'Unresolved cache link: {file}')
    shutil.copyfile(source, local_snapshot / file.name)
from safetensors import safe_open
for file in local_snapshot.glob('*.safetensors'):
    with safe_open(file, framework='pt', device='cpu') as weights:
        print(file.name, file.stat().st_size, len(weights.keys()), 'tensors verified', flush=True)
os.environ['HF_HUB_CACHE'] = str(local_cache)
os.environ['HF_HUB_OFFLINE'] = '1'

config = 'configs/qlora-v0.4-taxonomy-balanced.yaml'
import yaml
settings = yaml.safe_load(Path(config).read_text())
assert hashlib.sha256(Path(settings['data']['validation_file']).read_bytes()).hexdigest() == settings['data']['validation_file_sha256']

def preserve(source, name):
    destination = Path('/content') / name
    shutil.copy2(source, destination)
    with zipfile.ZipFile(destination) as saved:
        assert saved.testzip() is None
    shutil.copy2(destination, backup / name)
    print(name, hashlib.sha256(destination.read_bytes()).hexdigest(), flush=True)

base_run = 'qwen2.5-3b-murl-v0.4-base-control-validation'
run_logged([sys.executable, '-u', 'scripts/evaluate_qlora_adapter.py', '--config', config,
                '--run-id', base_run])
preserve(root / 'runs' / f'{base_run}.zip', 'qlora-v04-base-control-validation-results.zip')
run_logged([sys.executable, '-u', 'scripts/run_qlora_smoke.py', '--config', config])
model_dir = Path(settings['training']['output_dir'])
shutil.copy2(archive, model_dir / 'source.zip')
shutil.copy2('data/training/v0.4-sft-train.metadata.json', model_dir / 'dataset_metadata.json')
training_zip = shutil.make_archive('/content/murl-v04-training', 'zip', root_dir=root, base_dir=model_dir)
preserve(training_zip, 'qlora-v04-taxonomy-balanced-training-results.zip')
shutil.copytree(model_dir / 'adapter', backup / 'adapter', dirs_exist_ok=True)
run_logged([sys.executable, '-u', 'scripts/evaluate_qlora_adapter.py', '--config', config,
                '--adapter', str(model_dir / 'adapter')])
preserve(root / 'runs' / (settings['adapter_inference']['run_id'] + '.zip'),
         'qlora-v04-taxonomy-balanced-validation-results.zip')
print('V04 COMPLETE: all three archives verified and saved to Drive.', flush=True)
