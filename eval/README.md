# Eval Module (Latent-First)

This module adds an evaluation pipeline **separate from training** to analyze JEPA latents on KOCH.

## Install evaluation dependencies

Training remains driven by `requirements.txt`.
For eval, install the dedicated dependencies:

```bash
python -m pip install -r requirements-eval.txt
```

Notes:
- Python 3.10/3.11 uses `torch==2.2.1` (training parity).
- Python 3.12/3.13 automatically uses a newer compatible series.

## Run evaluation

From the repository root:

```bash
python eval/main.py \
  checkpoint.run_id=<RUN_SUBDIR> \
  checkpoint.local_dir=/path/to/stablewm_home \
  dataset.name=train__observation_images_laptop
```

Explicit option when checkpoint is not stored under `<local_dir>/<run_id>/`:

```bash
python eval/main.py \
  checkpoint.path=/absolute/path/to/lewm_epoch_45_object.ckpt \
  dataset.name=train__observation_images_laptop
```

### With Hugging Face fallback

If checkpoint is not available locally, the script can download it:

```bash
HF_TOKEN=hf_xxx python eval/main.py \
  checkpoint.repo_id=<HF_MODEL_REPO_ID> \
  checkpoint.run_id=<RUN_SUBDIR>
```

The script looks for an `_object.ckpt` file under `<run_id>/` and automatically picks the most recent one (highest epoch).

## Outputs

Artifacts are written to:

```text
eval_artifacts/<run_id>/
```

including:

- `latent_pca.csv`
- `latent_tsne.csv`
- `probe_metrics.json`
- `nearest_neighbors.csv`
- `figures/*.png`
- `report.md`

## Main Hydra parameters

- `checkpoint.repo_id`, `checkpoint.run_id`, `checkpoint.local_dir`, `checkpoint.filename_pattern`
- `dataset.name`, `dataset.cache_dir`, `dataset.frameskip`, `dataset.history_size`, `dataset.num_preds`
- `extraction.num_episodes`, `extraction.max_windows`, `extraction.batch_size`, `extraction.log_every_batches`, `extraction.device`, `extraction.seed`
- `projection.tsne.*`
- `neighbors.k`, `neighbors.num_anchors`
- `probe.test_size`, `probe.ridge_alpha`
- `output.artifact_root`, `output.run_id`
