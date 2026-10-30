"""Execute a QLoRA experiment after uploading its source ZIP and mounting Drive.

Defaults preserve the v0.5 run; ``MURL_*`` environment variables select a
later frozen configuration without duplicating artifact-handling code.
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
            print(line, end='', flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, args)


drive = Path('/content/drive/MyDrive/MURL')
assert drive.parent.is_dir(), 'Mount Drive before running'
os.environ['HF_HOME'] = str(drive / 'huggingface-cache')
archive = Path(os.environ.get('MURL_SOURCE_ARCHIVE', '/content/murl-v05-source.zip'))
root = Path(os.environ.get('MURL_SOURCE_ROOT', '/content/UIC-MURL-v05'))
assert not root.exists(), 'Use a fresh source directory'
with zipfile.ZipFile(archive) as source:
    assert source.testzip() is None
    source.extractall(root)
backup = drive / os.environ.get('MURL_DRIVE_RUN', 'runs/v0.5-moderated-taxonomy-balance')
backup.mkdir(parents=True, exist_ok=True)
shutil.copy2(archive, backup / archive.name)
os.chdir(root)
run_logged([sys.executable, '-m', 'pip', 'install', '-q',
            'transformers==5.20.1', 'accelerate==1.15.0', 'bitsandbytes==0.50.2',
            'peft==0.21.2', 'pyyaml'])

from huggingface_hub import snapshot_download
config = os.environ.get('MURL_CONFIG', 'configs/qlora-v0.5-moderated-taxonomy-balance.yaml')
import yaml
settings = yaml.safe_load(Path(config).read_text())
revision = settings['base_model']['revision']
model_id = settings['base_model']['id']
snapshot = Path(snapshot_download(model_id, revision=revision, token=False))
local_cache = Path(os.environ.get('MURL_LOCAL_CACHE', '/content/murl-hf-v05'))
local_snapshot = local_cache / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
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

assert hashlib.sha256(Path(settings['data']['validation_file']).read_bytes()).hexdigest() == settings['data']['validation_file_sha256']

def preserve(source, name):
    destination = Path('/content') / name
    shutil.copy2(source, destination)
    with zipfile.ZipFile(destination) as saved:
        assert saved.testzip() is None
    shutil.copy2(destination, backup / name)
    print(name, hashlib.sha256(destination.read_bytes()).hexdigest(), flush=True)

base_run = os.environ.get('MURL_BASE_RUN', 'qwen2.5-3b-murl-v0.5-base-control-validation')
run_logged([sys.executable, '-u', 'scripts/evaluate_qlora_adapter.py', '--config', config, '--run-id', base_run])
preserve(root / 'runs' / f'{base_run}.zip', os.environ.get('MURL_BASE_ARCHIVE', 'qlora-v05-base-control-validation-results.zip'))
run_logged([sys.executable, '-u', 'scripts/run_qlora_smoke.py', '--config', config])
model_dir = Path(settings['training']['output_dir'])
shutil.copy2(archive, model_dir / 'source.zip')
shutil.copy2('data/training/v0.5-sft-train.metadata.json', model_dir / 'dataset_metadata.json')
training_zip = shutil.make_archive(os.environ.get('MURL_TRAINING_STEM', '/content/murl-v05-training'), 'zip', root_dir=root, base_dir=model_dir)
preserve(training_zip, os.environ.get('MURL_TRAINING_ARCHIVE', 'qlora-v05-moderated-taxonomy-balance-training-results.zip'))
shutil.copytree(model_dir / 'adapter', backup / 'adapter', dirs_exist_ok=True)
run_logged([sys.executable, '-u', 'scripts/evaluate_qlora_adapter.py', '--config', config,
            '--adapter', str(model_dir / 'adapter')])
preserve(root / 'runs' / (settings['adapter_inference']['run_id'] + '.zip'),
         os.environ.get('MURL_VALIDATION_ARCHIVE', 'qlora-v05-moderated-taxonomy-balance-validation-results.zip'))
print('EXPERIMENT COMPLETE: all three archives verified and saved to Drive.', flush=True)
