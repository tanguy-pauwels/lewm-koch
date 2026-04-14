# le-wm-koch

Training boilerplate to run **LeWM** (JEPA + SIGReg) on an **HDF5** dataset hosted on **Hugging Face**, with execution inside a **GPU Docker** container.

This wrapper keeps upstream `le-wm` code in `src/` and injects cloud-oriented runtime adaptations:
- load secrets from `.env`
- download dataset files to the container local disk
- generate Hydra dataset config at runtime
- run `src/train.py` with Hydra overrides
- upload checkpoints to the Hugging Face Hub

## Why JEPA for robot arm control (SO_ARM101 / KOCH)

This project started as a practical way to test LeWM architecture behavior for robot-arm control scenarios, especially on KOCH / SO_ARM101-style data.

Main goals:
- build a reproducible training pipeline around LeWM
- study whether performance improves when adding more episodes from the same task but different viewpoints (for example `laptop` and `phone` captures from the same dataset)
- study the impact of adding non-robotic data on latent-space stability

## Useful links

- Eval module documentation: [`eval/README.md`](eval/README.md)
- Docker image: [tpauwels/le-wm-koch on Docker Hub](https://hub.docker.com/repository/docker/tpauwels/le-wm-koch/general)
- LeRobot dataset conversion pipeline (parquet + MP4 -> HDF5): [tanguy-pauwels/lerobot-dataset-to-HDF5](https://github.com/tanguy-pauwels/lerobot-dataset-to-HDF5/tree/main)
- Pod template for easier training: [le-wm-koch-public](https://console.runpod.io/deploy?template=f83357qr5r&ref=7x06vrca)

## Stack

- GPU base image: `nvcr.io/nvidia/pytorch:24.02-py3`
- World model: `lucas-maes/le-wm`
- Core library: `stable-worldmodel[train,env]`
- Tracking: Weights & Biases
- Artifact storage: Hugging Face Hub

## Repository structure

```text
le-wm-koch/
├── config/                 # Hydra overlays added by this repo
├── data/                   # Local HDF5 cache + run outputs if STABLEWM_HOME points here
├── eval/                   # Latent-first evaluation module (Hydra + viz + probes)
├── src/                    # Upstream le-wm code, kept intact
├── .env.example            # Configuration template
├── Dockerfile              # GPU training image
├── requirements.txt        # Pinned Python dependencies
├── requirements-eval.txt   # Evaluation-only dependencies (without impacting training image)
├── train_wrapper.py        # Main orchestrator
└── README.md
```

## Prerequisites

- Docker with NVIDIA GPU access
- A Hugging Face token with access to the source dataset
- Optional: a WandB token
- Recommended on H100/4090: `--shm-size=16g`

## Quick start

### 1. Prepare environment variables

```bash
cp .env.example .env
```

Minimum values to set:

```dotenv
HF_TOKEN=hf_xxx
WANDB_TOKEN=...
WANDB_ENTITY=...
WANDB_PROJECT=le-wm-koch
HF_OUTPUT_REPO_ID=your-username/lewm-koch-checkpoints
```

If you do not want WandB:

```dotenv
WANDB_ENABLED=false
```

### 2. Build the image

```bash
docker build -t lewm-koch .
```

### 3. Run training

```bash
docker run --rm \
  --gpus all \
  --shm-size=16g \
  --env-file .env \
  -v $(pwd)/data:/workspace/data \
  lewm-koch
```

What this command does:
- downloads `.h5` files from `HF_DATASET_REPO_ID` into `STABLEWM_HOME`
- generates `src/config/train/data/train__observation_images_merged.yaml` (if multi-dataset mode is enabled)
- runs `python src/train.py ... loader.batch_size=256 trainer.precision=bf16`
- automatically falls back to a safer profile (`batch_size`, `num_workers`, `prefetch_factor`) on CUDA OOM or process kill (`exit -9` / SIGKILL)
- writes artifacts to `STABLEWM_HOME/<RUN_SUBDIR>`
- pushes checkpoints to `HF_OUTPUT_REPO_ID` under `checkpoint-*` directories at each eval

## Useful commands

### Standard run

```bash
docker run --rm --gpus all --shm-size=16g --env-file .env \
  -v $(pwd)/data:/workspace/data \
  lewm-koch
```

### Wrapper dry-run

Useful to inspect actions without training or uploading.

```bash
docker run --rm --gpus all --env-file .env \
  -v $(pwd)/data:/workspace/data \
  tpauwels/le-wm-koch --dry-run --skip-train
```

### Reuse an already-downloaded dataset

If `.h5` files are already present in `STABLEWM_HOME`, skip download.

```bash
docker run --rm --gpus all --shm-size=16g --env-file .env \
  -v $(pwd)/data:/workspace/data \
  lewm-koch --skip-download
```

### Upload later only

```bash
docker run --rm --gpus all --shm-size=16g --env-file .env \
  -v $(pwd)/data:/workspace/data \
  tpauwels/le-wm-koch --skip-sync
```

### Local execution without Docker

Only if your local Python environment already has compatible GPU dependencies.

```bash
python3 train_wrapper.py
```

## Latent evaluation (separate module)

The latent-first JEPA evaluation pipeline lives in `eval/` and does not impact the training stack.

For full evaluation usage, see [`eval/README.md`](eval/README.md).

### Install eval dependencies

```bash
python -m pip install -r requirements-eval.txt
```

### Run a local evaluation (checkpoint already present)

```bash
python eval/main.py \
  checkpoint.run_id=<RUN_SUBDIR> \
  checkpoint.local_dir=/path/to/stablewm_home \
  dataset.name=train__observation_images_laptop
```

If the `.ckpt` is stored directly in a folder (without a `run_id` subfolder), use:

```bash
python eval/main.py \
  checkpoint.path=/absolute/path/to/lewm_epoch_45_object.ckpt \
  dataset.name=train__observation_images_laptop
```

### Run with Hugging Face fallback

```bash
HF_TOKEN=hf_xxx python eval/main.py \
  checkpoint.repo_id=<HF_MODEL_REPO_ID> \
  checkpoint.run_id=<RUN_SUBDIR>
```

Produced artifacts:
- `eval_artifacts/<run_id>/latent_pca.csv`
- `eval_artifacts/<run_id>/latent_tsne.csv`
- `eval_artifacts/<run_id>/probe_metrics.json`
- `eval_artifacts/<run_id>/figures/*.png`

## Environment variables

### Hugging Face

| Variable | Required | Role | Example |
| --- | --- | --- | --- |
| `HF_TOKEN` | yes for HF download/upload | Hugging Face token used to read dataset and push checkpoints | `hf_xxx` |
| `HF_DATASET_REPO_ID` | yes | Source dataset repository on the Hub | `Tpauwels/lerobot-hdf5-koch_pick_place_1_lego` |
| `HF_DATASET_PATTERNS` | no | Glob patterns used to list downloadable files | `*.h5` |
| `HF_DATASET_FILES` | no | CSV list of exact dataset-repo paths to download. If empty, wrapper uses `HF_DATASET_PATTERNS`. | `train/train__observation_images_laptop.h5,validation/val.h5` |
| `HF_FORCE_DOWNLOAD` | no | Force re-download even if file already exists in local cache | `true` |
| `HF_OUTPUT_REPO_ID` | no | Model repo where checkpoints are uploaded at end of run | `tanguy/lewm-koch-checkpoints` |
| `HF_OUTPUT_PRIVATE` | no | Create output repo as private if needed | `true` |
| `HF_UPLOAD_ALL_CHECKPOINTS` | no | If `true`, upload all `_object.ckpt` checkpoints. Otherwise upload only the latest, plus final weight and config files. | `false` |
| `HF_HUB_ENABLE_HF_TRANSFER` | no | Enable `hf_transfer` backend for faster HF downloads | `1` |

### `HF_DATASET_FILES` behavior

This is the most ambiguous variable, so here is the exact rule:

- Empty `HF_DATASET_FILES`: wrapper lists all files in the dataset repo and keeps those matching `HF_DATASET_PATTERNS`
- Non-empty `HF_DATASET_FILES`: wrapper **ignores** `HF_DATASET_PATTERNS` and downloads only listed paths

Example 1, simple dataset repo with one root `.h5`:

```dotenv
HF_DATASET_PATTERNS=*.h5
HF_DATASET_FILES=
```

Example 2, dataset repo with multiple splits, and you want one exact file:

```dotenv
HF_DATASET_FILES=train/train__observation_images_laptop.h5
```

Example 3, dataset repo with multiple explicit files:

```dotenv
HF_DATASET_FILES=train/part-000.h5,train/part-001.h5
```

Important:
- `HF_DATASET_FILES` values are **paths inside the Hugging Face repo**, not local paths
- the dataset name seen by `le-wm` remains the `.h5` stem, for example `train__observation_images_laptop`

### WandB

| Variable | Required | Role | Example |
| --- | --- | --- | --- |
| `WANDB_ENABLED` | no | Enable or disable WandB logger | `true` |
| `WANDB_TOKEN` | no but recommended if WandB enabled | WandB API token | `...` |
| `WANDB_ENTITY` | yes if WandB enabled | WandB workspace/team | `pauwelstanguy` |
| `WANDB_PROJECT` | yes if WandB enabled | WandB project | `le-wm-koch` |
| `WANDB_NAME` | no | Human-readable run name. If empty, wrapper uses `RUN_SUBDIR`. | `lewm-h100-run-01` |

### Training / local storage

| Variable | Required | Role | Example |
| --- | --- | --- | --- |
| `STABLEWM_HOME` | no | Local folder containing `.h5` files and run outputs | `/workspace/data` |
| `RUN_SUBDIR` | no | Run subfolder name. If empty, auto-generated | `lewm-koch-20260331-120000` |
| `LEWM_OUTPUT_MODEL_NAME` | no | Prefix for checkpoints generated by `src/train.py` | `lewm` |
| `LEWM_BATCH_SIZE` | no | Initial target batch size (attempt 1) | `256` |
| `LEWM_BATCH_SIZE_FALLBACK` | no | Fallback batch size on CUDA OOM or process kill (`-9`) | `128` |
| `LEWM_PRECISION` | no | Hydra override `trainer.precision` | `bf16` |
| `LEWM_NUM_WORKERS` | no | Hydra override `loader.num_workers` | `8` |
| `LEWM_LOADER_PREFETCH_FACTOR` | no | Hydra override `loader.prefetch_factor` | `2` |
| `LEWM_LOADER_PERSISTENT_WORKERS` | no | Hydra override `loader.persistent_workers` | `true` |
| `LEWM_FALLBACK_NUM_WORKERS` | no | `loader.num_workers` for fallback profile | `2` |
| `LEWM_FALLBACK_PREFETCH_FACTOR` | no | `loader.prefetch_factor` for fallback profile | `1` |
| `LEWM_FALLBACK_PERSISTENT_WORKERS` | no | `loader.persistent_workers` for fallback profile | `false` |
| `LEWM_SIGREG_WEIGHT` | no | Hydra override `loss.sigreg.weight` | `0.05` |
| `LEWM_EVAL_EVERY_N_EPOCHS` | no | Hydra override `trainer.check_val_every_n_epoch` | `10` |
| `LEWM_PUSH_CHECKPOINT_ON_EVAL` | no | Synchronous HF push at each eval | `true` |
| `LEWM_HF_CHECKPOINT_LAYOUT` | no | HF layout: `transformers`, `epochs`, `flat` | `transformers` |
| `LEWM_EXTRA_OVERRIDES` | no | CSV list of extra Hydra overrides injected as-is | `trainer.max_epochs=20,optimizer.lr=1e-4` |

### Runtime Hydra dataset generation

| Variable | Required | Role | Example |
| --- | --- | --- | --- |
| `LEWM_DATASET_NAMES` | no | CSV list of datasets to merge into one run | `train__observation_images_laptop,train__observation_images_phone` |
| `LEWM_DATASET_NAME` | no | Backward-compatible single-dataset mode | `train__observation_images_laptop` |
| `LEWM_DATA_CONFIG_NAME` | no | Name of YAML generated in `src/config/train/data/` | `train__observation_images_merged` |
| `LEWM_FRAME_SKIP` | no | Value injected in Hydra dataset config | `10` |
| `LEWM_KEYS_TO_LOAD` | no | HDF5 columns loaded | `pixels,action,proprio,state` |
| `LEWM_KEYS_TO_CACHE` | no | HDF5 columns cached on dataset side | `action,proprio,state` |

### Source bootstrap

| Variable | Required | Role | Example |
| --- | --- | --- | --- |
| `LEWM_SOURCE_REPO` | no | Git repo cloned if `src/` is missing | `https://github.com/lucas-maes/le-wm.git` |
| `LEWM_SOURCE_REF` | no | Commit or tag checked out after clone | `ca231f9f9d9ab041034b6d05e90b6e04bd6cff82` |

## Hydra overrides injected at runtime

The wrapper runs `src/train.py` with at least:

```bash
python src/train.py \
  data=train__observation_images_merged \
  subdir=<RUN_SUBDIR> \
  output_model_name=lewm \
  loader.batch_size=128 \
  loader.num_workers=8 \
  loader.prefetch_factor=2 \
  loader.persistent_workers=true \
  trainer.precision=bf16 \
  trainer.check_val_every_n_epoch=10 \
  checkpoint.object_epoch_interval=10 \
  loss.sigreg.weight=0.05
```

If WandB is enabled, it also adds:

```bash
wandb.enabled=true \
wandb.config.entity=<WANDB_ENTITY> \
wandb.config.project=<WANDB_PROJECT> \
wandb.config.name=<WANDB_NAME or RUN_SUBDIR> \
wandb.config.id=<RUN_SUBDIR> \
wandb.config.resume=allow
```

## Produced artifacts

In `STABLEWM_HOME/<RUN_SUBDIR>/`, you can find:
- `config.yaml`: resolved Hydra config for the run
- `run_manifest.json`: wrapper-generated manifest
- `lewm_weights.ckpt`: Lightning weight checkpoint
- `lewm_epoch_<n>_object.ckpt`: LeWM object checkpoints

If `HF_OUTPUT_REPO_ID` is set, these files are uploaded under:

```text
<HF_OUTPUT_REPO_ID>/<RUN_SUBDIR>/checkpoints/checkpoint-00010/...
```

## Cloud GPU recommendations

- H100 / 4090: `--shm-size=16g`
- Tune `LEWM_NUM_WORKERS` between `4` and `12` based on CPU, then increase only if GPU remains underutilized
- Mount a persistent volume on `/workspace/data` to avoid re-downloading `.h5` files every run
- If you see `Training failed with exit code -9`, this is usually host-memory pressure: reduce `LEWM_NUM_WORKERS` and/or keep aggressive fallback (`LEWM_FALLBACK_NUM_WORKERS=2`, `LEWM_FALLBACK_PREFETCH_FACTOR=1`)
- Avoid `HF_UPLOAD_ALL_CHECKPOINTS=true` on long runs unless you explicitly need all epochs archived

## Quick troubleshooting

### Wrapper says it does not know which dataset to use

Set explicit multi-dataset mode:

```dotenv
LEWM_DATASET_NAMES=train__observation_images_laptop,train__observation_images_phone
```

Single-dataset mode (backward compatible):

```text
LEWM_DATASET_NAME=train__observation_images_laptop
```

### Wrapper downloads too many files

Do not leave only `HF_DATASET_PATTERNS=*.h5` if your dataset repo contains multiple unwanted `.h5` files.
Use instead:

```dotenv
HF_DATASET_FILES=train/train__observation_images_laptop.h5
```

### Disable WandB

```dotenv
WANDB_ENABLED=false
```

### Force a stable run name

```dotenv
RUN_SUBDIR=lewm-koch-h100-test-01
WANDB_NAME=lewm-koch-h100-test-01
```

## Credits and acknowledgments

- **LeWM model**: based on [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) by Lucas Maes and contributors.
- **KOCH / LeRobot datasets**: thanks to dataset creators and maintainers in the LeRobot and KOCH ecosystem.

## License

This repository is distributed under Apache 2.0 terms, with an explicit MIT notice for upstream `le-wm` code preserved in `src/`.
See [`LICENSE.md`](LICENSE.md).
