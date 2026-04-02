# Eval Module (Latent-First)

Ce module ajoute une pipeline d'evaluation **separee du training** pour analyser les latents JEPA sur KOCH.

## Installation des dependances eval

Le training reste pilote par `requirements.txt`.
Pour l'eval, installe les dependances dediees:

```bash
python -m pip install -r requirements-eval.txt
```

Note:
- Python 3.10/3.11 utilise `torch==2.2.1` (parite training).
- Python 3.12/3.13 utilise automatiquement une serie plus recente compatible.

## Lancer l'evaluation

Depuis la racine du repo:

```bash
python eval/main.py \
  checkpoint.run_id=<RUN_SUBDIR> \
  checkpoint.local_dir=/path/to/stablewm_home \
  dataset.name=train__observation_images_laptop
```

Option explicite si le checkpoint n'est pas range sous `<local_dir>/<run_id>/`:

```bash
python eval/main.py \
  checkpoint.path=/absolute/path/to/lewm_epoch_45_object.ckpt \
  dataset.name=train__observation_images_laptop
```

### Avec fallback Hugging Face

Si le checkpoint n'est pas present localement, le script peut le telecharger:

```bash
HF_TOKEN=hf_xxx python eval/main.py \
  checkpoint.repo_id=<HF_MODEL_REPO_ID> \
  checkpoint.run_id=<RUN_SUBDIR>
```

Le script cherche un fichier `_object.ckpt` sous `<run_id>/` et choisit automatiquement le plus recent (epoch la plus elevee).

## Sorties

Les artefacts sont ecrits dans:

```text
eval_artifacts/<run_id>/
```

avec notamment:

- `latent_pca.csv`
- `latent_tsne.csv`
- `probe_metrics.json`
- `nearest_neighbors.csv`
- `figures/*.png`
- `report.md`

## Parametres Hydra principaux

- `checkpoint.repo_id`, `checkpoint.run_id`, `checkpoint.local_dir`, `checkpoint.filename_pattern`
- `dataset.name`, `dataset.cache_dir`, `dataset.frameskip`, `dataset.history_size`, `dataset.num_preds`
- `extraction.num_episodes`, `extraction.max_windows`, `extraction.batch_size`, `extraction.log_every_batches`, `extraction.device`, `extraction.seed`
- `projection.tsne.*`
- `neighbors.k`, `neighbors.num_anchors`
- `probe.test_size`, `probe.ridge_alpha`
- `output.artifact_root`, `output.run_id`
