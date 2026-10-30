"""Execute the precommitted v0.7 two-stage diagnosis experiment in Colab."""
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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


drive = Path('/content/drive/MyDrive/MURL')
assert drive.parent.is_dir(), 'Mount Drive before running'
os.environ['HF_HOME'] = str(drive / 'huggingface-cache')
archive = Path('/content/murl-v07-source.zip')
root = Path('/content/UIC-MURL-v07')
assert not root.exists(), 'Use a fresh source directory'
with zipfile.ZipFile(archive) as source:
    assert source.testzip() is None
    source.extractall(root)
backup = drive / 'runs/v0.7-two-stage-diagnosis'
backup.mkdir(parents=True, exist_ok=True)
source_backup = backup / archive.name
shutil.copy2(archive, source_backup)
assert file_sha256(archive) == file_sha256(source_backup)
print('source', file_sha256(archive), flush=True)
os.chdir(root)
run_logged([sys.executable, '-m', 'pip', 'install', '-q',
            'transformers==5.20.1', 'accelerate==1.15.0', 'bitsandbytes==0.50.2',
            'peft==0.21.2', 'pyyaml'])

from huggingface_hub import snapshot_download
import yaml
config = 'configs/qlora-v0.7-two-stage-diagnosis.yaml'
settings = yaml.safe_load(Path(config).read_text())
revision = settings['base_model']['revision']
model_id = settings['base_model']['id']
snapshot = Path(snapshot_download(model_id, revision=revision, token=False))
local_cache = Path('/content/murl-hf-v07')
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

assert file_sha256(settings['data']['validation_file']) == settings['data']['validation_file_sha256']


def preserve(source, name):
    destination = Path('/content') / name
    shutil.copy2(source, destination)
    with zipfile.ZipFile(destination) as saved:
        assert saved.testzip() is None
    drive_destination = backup / name
    shutil.copy2(destination, drive_destination)
    local_hash = file_sha256(destination)
    drive_hash = file_sha256(drive_destination)
    assert local_hash == drive_hash
    print(name, local_hash, 'local/Drive match', flush=True)


base_run = 'qwen2.5-3b-murl-v0.7-base-control-validation'
run_logged([sys.executable, '-u', 'scripts/evaluate_qlora_adapter.py',
            '--config', config, '--run-id', base_run])
preserve(root / 'runs' / f'{base_run}.zip',
         'qlora-v07-base-control-validation-results.zip')
run_logged([sys.executable, '-u', 'scripts/run_qlora_smoke.py', '--config', config])
model_dir = Path(settings['training']['output_dir'])
shutil.copy2(archive, model_dir / 'source.zip')
shutil.copy2('data/training/v0.5-sft-train.metadata.json', model_dir / 'dataset_metadata.json')
training_zip = shutil.make_archive('/content/murl-v07-training', 'zip', root_dir=root, base_dir=model_dir)
preserve(training_zip, 'qlora-v07-two-stage-diagnosis-training-results.zip')
shutil.copytree(model_dir / 'checkpoint-30', backup / 'stage-1-checkpoint', dirs_exist_ok=True)
shutil.copytree(model_dir / 'adapter', backup / 'final-adapter', dirs_exist_ok=True)
run_logged([sys.executable, '-u', 'scripts/evaluate_qlora_adapter.py', '--config', config,
            '--adapter', str(model_dir / 'adapter')])
preserve(root / 'runs' / (settings['adapter_inference']['run_id'] + '.zip'),
         'qlora-v07-two-stage-diagnosis-validation-results.zip')
print('EXPERIMENT COMPLETE: all three archives verified and saved to Drive.', flush=True)
