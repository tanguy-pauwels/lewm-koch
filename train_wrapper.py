#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
SRC_TRAIN_SCRIPT = SRC_DIR / 'train.py'
LOCAL_CONFIG_DIR = ROOT_DIR / 'config'
SRC_CONFIG_DIR = SRC_DIR / 'config'
DEFAULT_SOURCE_REPO = 'https://github.com/lucas-maes/le-wm.git'
DEFAULT_SOURCE_REF = 'ca231f9f9d9ab041034b6d05e90b6e04bd6cff82'
DEFAULT_DATASET_REPO = 'Tpauwels/lerobot-hdf5-koch_pick_place_1_lego'
DEFAULT_DATASET_NAME = 'train__observation_images_laptop'
DEFAULT_DATA_CONFIG = 'train__observation_images_laptop'
DEFAULT_OUTPUT_MODEL_NAME = 'lewm'
DEFAULT_BATCH_SIZE = 128
DEFAULT_PRECISION = 'bf16'
DEFAULT_FRAME_SKIP = 5
DEFAULT_KEYS_TO_LOAD = ('pixels', 'action', 'proprio', 'state')
DEFAULT_KEYS_TO_CACHE = ('action', 'proprio', 'state')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download HDF5 data, launch le-wm training, and sync checkpoints to the Hugging Face Hub.'
    )
    parser.add_argument('--run-subdir', default=os.getenv('RUN_SUBDIR'))
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-sync', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def parse_csv(value: str | None, *, default: Sequence[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(',') if item.strip()]


def load_environment() -> None:
    load_dotenv(ROOT_DIR / '.env')
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('HYDRA_FULL_ERROR', '1')

    stablewm_home = Path(os.getenv('STABLEWM_HOME', ROOT_DIR / 'data')).expanduser().resolve()
    stablewm_home.mkdir(parents=True, exist_ok=True)
    os.environ['STABLEWM_HOME'] = str(stablewm_home)

    wandb_token = os.getenv('WANDB_TOKEN')
    if wandb_token:
        os.environ.setdefault('WANDB_API_KEY', wandb_token)


def log_runtime_versions() -> None:
    packages = [
        'torch',
        'torchvision',
        'transformers',
        'lightning',
        'stable-worldmodel',
        'stable-pretraining',
        'huggingface_hub',
        'torchmetrics',
    ]
    print(f'[env] python={sys.version.split()[0]}', flush=True)
    for package in packages:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = 'MISSING'
        print(f'[env] {package}={version}', flush=True)


class CommandError(RuntimeError):
    pass


class TrainingError(RuntimeError):
    def __init__(self, returncode: int):
        super().__init__(f'Training failed with exit code {returncode}.')
        self.returncode = returncode


def run_command(command: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print('$', ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ensure_source_tree() -> None:
    if SRC_TRAIN_SCRIPT.exists():
        return

    repo_url = os.getenv('LEWM_SOURCE_REPO', DEFAULT_SOURCE_REPO)
    repo_ref = os.getenv('LEWM_SOURCE_REF', DEFAULT_SOURCE_REF)

    print(f'[bootstrap] src/ missing, cloning {repo_url} at {repo_ref}', flush=True)
    with TemporaryDirectory(prefix='lewm-src-') as tmpdir:
        tmp_path = Path(tmpdir) / 'le-wm'
        run_command(['git', 'clone', repo_url, str(tmp_path)])
        if repo_ref:
            run_command(['git', 'checkout', repo_ref], cwd=tmp_path)
        shutil.copytree(tmp_path, SRC_DIR, dirs_exist_ok=True)
    shutil.rmtree(SRC_DIR / '.git', ignore_errors=True)

    if not SRC_TRAIN_SCRIPT.exists():
        raise CommandError(f'Missing expected training entrypoint: {SRC_TRAIN_SCRIPT}')


def sync_local_config() -> None:
    if not LOCAL_CONFIG_DIR.exists():
        return
    shutil.copytree(LOCAL_CONFIG_DIR, SRC_CONFIG_DIR, dirs_exist_ok=True)


def list_existing_hdf5_files(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.rglob('*.h5') if path.is_file())


def list_remote_dataset_files(repo_id: str, token: str | None, patterns: Sequence[str]) -> list[str]:
    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
    selected = [
        file_name
        for file_name in sorted(repo_files)
        if any(fnmatch.fnmatch(file_name, pattern) for pattern in patterns)
    ]
    if not selected:
        raise CommandError(
            f'No remote dataset files in {repo_id!r} matched patterns: {", ".join(patterns)}'
        )
    return selected


def download_data() -> list[Path]:
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    repo_id = os.getenv('HF_DATASET_REPO_ID', DEFAULT_DATASET_REPO)
    patterns = parse_csv(os.getenv('HF_DATASET_PATTERNS'), default=('*.h5',))
    data_root = Path(os.environ['STABLEWM_HOME'])

    requested_files = parse_csv(os.getenv('HF_DATASET_FILES'), default=())
    if requested_files:
        remote_files = requested_files
    else:
        remote_files = list_remote_dataset_files(repo_id=repo_id, token=token, patterns=patterns)

    downloaded_files: list[Path] = []
    force_download = parse_bool(os.getenv('HF_FORCE_DOWNLOAD'), default=False)
    print(
        f'[data] downloading {len(remote_files)} file(s) from {repo_id} into {data_root} '
        f'(hf_transfer={os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0")})',
        flush=True,
    )
    for remote_file in remote_files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=remote_file,
            local_dir=data_root,
            token=token,
            force_download=force_download,
        )
        downloaded_files.append(Path(local_path).resolve())
        print(f'[data] ready: {remote_file} -> {local_path}', flush=True)
    return downloaded_files


def resolve_dataset_name(local_files: Sequence[Path]) -> str:
    dataset_name = os.getenv('LEWM_DATASET_NAME')
    if dataset_name:
        return dataset_name

    stems = sorted({path.stem for path in local_files if path.suffix == '.h5'})
    if DEFAULT_DATASET_NAME in stems:
        return DEFAULT_DATASET_NAME
    if len(stems) == 1:
        return stems[0]
    if not stems:
        raise CommandError('No local HDF5 file found under STABLEWM_HOME and LEWM_DATASET_NAME was not provided.')
    raise CommandError(
        'Multiple HDF5 files found. Set LEWM_DATASET_NAME to the file stem you want to train on. '
        f'Found: {", ".join(stems)}'
    )


def write_runtime_data_config(dataset_name: str) -> str:
    data_config_name = os.getenv('LEWM_DATA_CONFIG_NAME', DEFAULT_DATA_CONFIG)
    frame_skip = int(os.getenv('LEWM_FRAME_SKIP', str(DEFAULT_FRAME_SKIP)))
    keys_to_load = parse_csv(os.getenv('LEWM_KEYS_TO_LOAD'), default=DEFAULT_KEYS_TO_LOAD)
    keys_to_cache = parse_csv(os.getenv('LEWM_KEYS_TO_CACHE'), default=DEFAULT_KEYS_TO_CACHE)

    target_dir = SRC_CONFIG_DIR / 'train' / 'data'
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f'{data_config_name}.yaml'

    load_block = '\n'.join(f'    - {key}' for key in keys_to_load)
    cache_block = '\n'.join(f'    - {key}' for key in keys_to_cache)
    target_file.write_text(
        '\n'.join(
            [
                'dataset:',
                "  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}",
                f'  frameskip: {frame_skip}',
                f'  name: {dataset_name}',
                '  keys_to_load:',
                load_block,
                '  keys_to_cache:',
                cache_block,
                '',
            ]
        ),
        encoding='utf-8',
    )
    print(f'[config] wrote runtime dataset config: {target_file}', flush=True)
    return data_config_name


def should_enable_wandb() -> bool:
    explicit = os.getenv('WANDB_ENABLED')
    if explicit is not None:
        return parse_bool(explicit, default=True)
    return bool(os.getenv('WANDB_ENTITY') and os.getenv('WANDB_PROJECT'))


def build_hydra_overrides(run_subdir: str, data_config_name: str) -> list[str]:
    batch_size = int(os.getenv('LEWM_BATCH_SIZE', str(DEFAULT_BATCH_SIZE)))
    precision = os.getenv('LEWM_PRECISION', DEFAULT_PRECISION)
    output_model_name = os.getenv('LEWM_OUTPUT_MODEL_NAME', DEFAULT_OUTPUT_MODEL_NAME)
    cpu_count = os.cpu_count() or 8
    num_workers_default = max(4, min(16, cpu_count // 2))
    num_workers = int(os.getenv('LEWM_NUM_WORKERS', str(num_workers_default)))

    overrides = [
        f'data={data_config_name}',
        f'subdir={run_subdir}',
        f'output_model_name={output_model_name}',
        f'loader.batch_size={batch_size}',
        f'loader.num_workers={num_workers}',
        f'trainer.precision={precision}',
    ]

    if should_enable_wandb():
        entity = os.getenv('WANDB_ENTITY')
        project = os.getenv('WANDB_PROJECT')
        name = os.getenv('WANDB_NAME') or run_subdir
        if not entity or not project:
            raise CommandError('WANDB_ENABLED is true but WANDB_ENTITY or WANDB_PROJECT is missing.')
        overrides.extend(
            [
                'wandb.enabled=true',
                f'wandb.config.entity={entity}',
                f'wandb.config.project={project}',
                f'wandb.config.name={name}',
                f'wandb.config.id={run_subdir}',
                'wandb.config.resume=allow',
            ]
        )
    else:
        overrides.append('wandb.enabled=false')

    extra_overrides = parse_csv(os.getenv('LEWM_EXTRA_OVERRIDES'), default=())
    overrides.extend(extra_overrides)
    return overrides


def write_run_manifest(run_dir: Path, dataset_files: Sequence[Path], hydra_overrides: Sequence[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_repo_id': os.getenv('HF_DATASET_REPO_ID', DEFAULT_DATASET_REPO),
        'dataset_files': [str(path) for path in dataset_files],
        'hydra_overrides': list(hydra_overrides),
        'stablewm_home': os.environ['STABLEWM_HOME'],
    }
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def launch_training(run_subdir: str, hydra_overrides: Sequence[str], *, dry_run: bool) -> None:
    command = [sys.executable, str(SRC_TRAIN_SCRIPT), *hydra_overrides]
    if dry_run:
        print('[dry-run] training command:', ' '.join(command), flush=True)
        return

    env = os.environ.copy()
    pythonpath_entries = [str(ROOT_DIR)]
    existing_pythonpath = env.get('PYTHONPATH')
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_entries)
    print(f'[train] launching run {run_subdir}', flush=True)
    completed = subprocess.run(command, cwd=SRC_DIR, env=env)
    if completed.returncode != 0:
        raise TrainingError(completed.returncode)


def collect_checkpoint_files(run_dir: Path, output_model_name: str, upload_all: bool) -> list[Path]:
    config_file = run_dir / 'config.yaml'
    manifest_file = run_dir / 'run_manifest.json'
    weights_file = run_dir / f'{output_model_name}_weights.ckpt'
    object_files = sorted(run_dir.glob(f'{output_model_name}_epoch_*_object.ckpt'))
    checkpoint_files: list[Path] = []

    if weights_file.exists():
        checkpoint_files.append(weights_file)

    if upload_all:
        checkpoint_files.extend(object_files)
    elif object_files:
        checkpoint_files.append(object_files[-1])

    if not checkpoint_files:
        return []

    metadata_files = [path for path in (config_file, manifest_file) if path.exists()]
    return metadata_files + checkpoint_files


def sync_checkpoints(run_subdir: str, *, dry_run: bool) -> None:
    repo_id = os.getenv('HF_OUTPUT_REPO_ID')
    if not repo_id:
        print('[sync] HF_OUTPUT_REPO_ID not set, skipping checkpoint upload', flush=True)
        return

    output_model_name = os.getenv('LEWM_OUTPUT_MODEL_NAME', DEFAULT_OUTPUT_MODEL_NAME)
    upload_all = parse_bool(os.getenv('HF_UPLOAD_ALL_CHECKPOINTS'), default=False)
    run_dir = Path(os.environ['STABLEWM_HOME']) / run_subdir

    files_to_upload = collect_checkpoint_files(run_dir, output_model_name, upload_all=upload_all)
    if not files_to_upload:
        print(f'[sync] no checkpoint artifacts found under {run_dir}', flush=True)
        return

    api = None
    if not dry_run:
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if not token:
            raise CommandError('HF_OUTPUT_REPO_ID is set but HF_TOKEN is missing.')
        private_repo = parse_bool(os.getenv('HF_OUTPUT_PRIVATE'), default=False)
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type='model', private=private_repo, exist_ok=True)

    print(f'[sync] uploading {len(files_to_upload)} artifact(s) to {repo_id}', flush=True)
    for local_file in files_to_upload:
        repo_path = f'{run_subdir}/{local_file.name}'
        if dry_run:
            print(f'[dry-run] upload {local_file} -> {repo_id}:{repo_path}', flush=True)
            continue
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type='model',
        )
        print(f'[sync] uploaded {local_file.name}', flush=True)


def make_run_subdir() -> str:
    explicit = os.getenv('RUN_SUBDIR')
    if explicit:
        return explicit
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'lewm-koch-{timestamp}'


def main() -> int:
    try:
        args = parse_args()
        load_environment()
        log_runtime_versions()
        ensure_source_tree()
        sync_local_config()

        data_root = Path(os.environ['STABLEWM_HOME'])
        dataset_files: list[Path] = []
        if args.skip_download:
            dataset_files = list_existing_hdf5_files(data_root)
            print(f'[data] skip-download enabled, found {len(dataset_files)} local HDF5 file(s)', flush=True)
        else:
            dataset_files = download_data()

        dataset_name = resolve_dataset_name(dataset_files)
        data_config_name = write_runtime_data_config(dataset_name)

        run_subdir = args.run_subdir or make_run_subdir()
        os.environ['RUN_SUBDIR'] = run_subdir
        hydra_overrides = build_hydra_overrides(run_subdir, data_config_name)
        run_dir = data_root / run_subdir
        write_run_manifest(run_dir, dataset_files, hydra_overrides)

        training_error: TrainingError | None = None
        if not args.skip_train:
            try:
                launch_training(run_subdir, hydra_overrides, dry_run=args.dry_run)
            except TrainingError as exc:
                training_error = exc
                print(f'[train] {exc}', flush=True)

        if not args.skip_sync:
            sync_checkpoints(run_subdir, dry_run=args.dry_run)

        if training_error is not None:
            return training_error.returncode
        return 0
    except CommandError as exc:
        print(f'[error] {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
