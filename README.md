# le-wm-koch

Boilerplate de training pour entraîner **LeWM** (JEPA + SIGReg) sur un dataset **HDF5** hébergé sur **Hugging Face**, avec exécution dans un container **Docker GPU**.

Le wrapper garde le code upstream de `le-wm` dans `src/` et injecte les adaptations cloud au runtime:
- chargement des secrets depuis `.env`
- téléchargement du dataset vers le disque local du container
- génération de la config Hydra dataset
- lancement de `src/train.py` avec overrides Hydra
- upload final des checkpoints vers le Hugging Face Hub

## Stack

- Base GPU: `nvcr.io/nvidia/pytorch:24.02-py3`
- World model: `lucas-maes/le-wm`
- Librairie socle: `stable-worldmodel[train,env]`
- Tracking: Weights & Biases
- Stockage artefacts: Hugging Face Hub

## Structure

```text
le-wm-koch/
├── config/                 # Overlays Hydra ajoutés par ce repo
├── data/                   # Cache local HDF5 + sorties de run si STABLEWM_HOME pointe ici
├── eval/                   # Module d'évaluation latent-first (Hydra + viz + probes)
├── src/                    # Code upstream de le-wm, conservé intact
├── .env.example            # Template de configuration
├── Dockerfile              # Image d'entraînement GPU
├── requirements.txt        # Dépendances Python pinées
├── requirements-eval.txt   # Dépendances dédiées à l'évaluation (sans impacter l'image train)
├── train_wrapper.py        # Orchestrateur principal
└── README.md
```

## Pré-requis

- Docker avec accès GPU NVIDIA
- Un token Hugging Face avec accès au dataset source
- Optionnel: un token WandB
- Recommandé pour H100/4090: `--shm-size=16g`

## Démarrage rapide

### 1. Préparer les variables d'environnement

```bash
cp .env.example .env
```

Valeurs minimales à renseigner:

```dotenv
HF_TOKEN=hf_xxx
WANDB_TOKEN=...
WANDB_ENTITY=...
WANDB_PROJECT=le-wm-koch
HF_OUTPUT_REPO_ID=your-username/lewm-koch-checkpoints
```

Si tu ne veux pas utiliser WandB, mets:

```dotenv
WANDB_ENABLED=false
```

### 2. Builder l'image

```bash
docker build -t lewm-koch .
```

### 3. Lancer un entraînement

```bash
docker run --rm \
  --gpus all \
  --shm-size=16g \
  --env-file .env \
  -v $(pwd)/data:/workspace/data \
  lewm-koch
```

Ce que fait cette commande:
- télécharge les `.h5` depuis `HF_DATASET_REPO_ID` dans `STABLEWM_HOME`
- génère `src/config/train/data/train__observation_images_laptop.yaml`
- lance `python src/train.py ... loader.batch_size=64 trainer.precision=bf16`
- écrit les artefacts dans `STABLEWM_HOME/<RUN_SUBDIR>`
- pousse les artefacts sur `HF_OUTPUT_REPO_ID` si configuré

## Commandes utiles

### Run standard

```bash
docker run --rm --gpus all --shm-size=16g --env-file .env \
  -v $(pwd)/data:/workspace/data \
  lewm-koch
```

### Dry-run du wrapper

Utile pour voir ce qu'il va faire sans entraîner ni uploader.

```bash
docker run --rm --gpus all --env-file .env \
  -v $(pwd)/data:/workspace/data \
  tpauwels/le-wm-koch --dry-run --skip-train
```

### Réutiliser un dataset déjà téléchargé

Si les `.h5` sont déjà présents dans `STABLEWM_HOME`, tu peux sauter le download.

```bash
docker run --rm --gpus all --shm-size=16g --env-file .env \
  -v $(pwd)/data:/workspace/data \
  lewm-koch --skip-download
```

### N'uploader que plus tard

```bash
docker run --rm --gpus all --shm-size=16g --env-file .env \
  -v $(pwd)/data:/workspace/data \
  tpauwels/le-wm-koch --skip-sync
```

### Exécution locale hors Docker

Seulement si ton environnement Python local contient les dépendances GPU adaptées.

```bash
python3 train_wrapper.py
```

## Évaluation latente (module séparé)

Le pipeline d'évaluation JEPA latent-first est dans `eval/` et n'impacte pas la stack d'entraînement.

### Installer les dépendances eval

```bash
python -m pip install -r requirements-eval.txt
```

### Lancer une évaluation locale (checkpoint déjà présent)

```bash
python eval/main.py \
  checkpoint.run_id=<RUN_SUBDIR> \
  checkpoint.local_dir=/path/to/stablewm_home \
  dataset.name=train__observation_images_laptop
```

Si le `.ckpt` est stocke directement dans un dossier (sans sous-dossier `run_id`), utilise:

```bash
python eval/main.py \
  checkpoint.path=/absolute/path/to/lewm_epoch_45_object.ckpt \
  dataset.name=train__observation_images_laptop
```

### Lancer avec fallback Hugging Face

```bash
HF_TOKEN=hf_xxx python eval/main.py \
  checkpoint.repo_id=<HF_MODEL_REPO_ID> \
  checkpoint.run_id=<RUN_SUBDIR>
```

Artefacts produits:
- `eval_artifacts/<run_id>/latent_pca.csv`
- `eval_artifacts/<run_id>/latent_tsne.csv`
- `eval_artifacts/<run_id>/probe_metrics.json`
- `eval_artifacts/<run_id>/figures/*.png`

## Variables d'environnement

### Hugging Face

| Variable | Obligatoire | Rôle | Exemple |
| --- | --- | --- | --- |
| `HF_TOKEN` | oui si download ou upload HF | Token Hugging Face utilisé pour lire le dataset et pousser les checkpoints | `hf_xxx` |
| `HF_DATASET_REPO_ID` | oui | Repo dataset source sur le Hub | `Tpauwels/lerobot-hdf5-koch_pick_place_1_lego` |
| `HF_DATASET_PATTERNS` | non | Patterns glob utilisés pour lister les fichiers à télécharger | `*.h5` |
| `HF_DATASET_FILES` | non | Liste CSV de chemins exacts à télécharger dans le repo dataset. Si vide, le wrapper utilise `HF_DATASET_PATTERNS`. | `train/train__observation_images_laptop.h5,validation/val.h5` |
| `HF_FORCE_DOWNLOAD` | non | Force le re-download même si le fichier est déjà présent dans le cache local | `true` |
| `HF_OUTPUT_REPO_ID` | non | Repo modèle où uploader les checkpoints de fin de run | `tanguy/lewm-koch-checkpoints` |
| `HF_OUTPUT_PRIVATE` | non | Crée le repo output en privé si besoin | `true` |
| `HF_UPLOAD_ALL_CHECKPOINTS` | non | Si `true`, upload tous les checkpoints `_object.ckpt`. Sinon, upload le dernier seulement, plus le poids final et les fichiers de config | `false` |
| `HF_HUB_ENABLE_HF_TRANSFER` | non | Active le backend `hf_transfer` pour accélérer les downloads depuis HF | `1` |

### Explication de `HF_DATASET_FILES`

C'est la variable la plus ambiguë, donc voici la règle exacte:

- `HF_DATASET_FILES` vide: le wrapper liste tous les fichiers du repo dataset et garde ceux qui matchent `HF_DATASET_PATTERNS`
- `HF_DATASET_FILES` renseigné: le wrapper **ignore** `HF_DATASET_PATTERNS` et télécharge uniquement les chemins indiqués

Exemple 1, repo dataset simple avec un seul `.h5` à la racine:

```dotenv
HF_DATASET_PATTERNS=*.h5
HF_DATASET_FILES=
```

Exemple 2, repo dataset avec plusieurs splits et on veut un seul fichier précis:

```dotenv
HF_DATASET_FILES=train/train__observation_images_laptop.h5
```

Exemple 3, repo dataset avec plusieurs fichiers à prendre explicitement:

```dotenv
HF_DATASET_FILES=train/part-000.h5,train/part-001.h5
```

Important:
- les valeurs de `HF_DATASET_FILES` sont des **chemins internes au repo Hugging Face**, pas des chemins locaux
- le nom du dataset vu par `le-wm` reste le **stem** du fichier `.h5`, par exemple `train__observation_images_laptop`

### WandB

| Variable | Obligatoire | Rôle | Exemple |
| --- | --- | --- | --- |
| `WANDB_ENABLED` | non | Active ou non le logger WandB | `true` |
| `WANDB_TOKEN` | non mais recommandé si WandB actif | Token API WandB | `...` |
| `WANDB_ENTITY` | oui si WandB actif | Workspace ou équipe WandB | `pauwelstanguy` |
| `WANDB_PROJECT` | oui si WandB actif | Projet WandB | `le-wm-koch` |
| `WANDB_NAME` | non | Nom lisible du run. Si vide, le wrapper prend `RUN_SUBDIR` | `lewm-h100-run-01` |

### Training / stockage local

| Variable | Obligatoire | Rôle | Exemple |
| --- | --- | --- | --- |
| `STABLEWM_HOME` | non | Dossier local contenant les `.h5` et les sorties de run | `/workspace/data` |
| `RUN_SUBDIR` | non | Sous-dossier du run. Si vide, généré automatiquement | `lewm-koch-20260331-120000` |
| `LEWM_OUTPUT_MODEL_NAME` | non | Préfixe des checkpoints générés par `src/train.py` | `lewm` |
| `LEWM_BATCH_SIZE` | non | Override Hydra `loader.batch_size` | `64` |
| `LEWM_PRECISION` | non | Override Hydra `trainer.precision` | `bf16` |
| `LEWM_NUM_WORKERS` | non | Override Hydra `loader.num_workers` | `16` |
| `LEWM_EXTRA_OVERRIDES` | non | Liste CSV d'overrides Hydra supplémentaires injectés tels quels | `trainer.max_epochs=20,optimizer.lr=1e-4` |

### Dataset Hydra runtime

| Variable | Obligatoire | Rôle | Exemple |
| --- | --- | --- | --- |
| `LEWM_DATASET_NAME` | souvent oui | Nom du dataset attendu par `le-wm`, donc le stem du `.h5` sans extension | `train__observation_images_laptop` |
| `LEWM_DATA_CONFIG_NAME` | non | Nom du fichier YAML généré dans `src/config/train/data/` | `train__observation_images_laptop` |
| `LEWM_FRAME_SKIP` | non | Valeur injectée dans la config Hydra dataset | `5` |
| `LEWM_KEYS_TO_LOAD` | non | Colonnes HDF5 chargées | `pixels,action,proprio,state` |
| `LEWM_KEYS_TO_CACHE` | non | Colonnes gardées en cache côté dataset | `action,proprio,state` |

### Bootstrap du code source

| Variable | Obligatoire | Rôle | Exemple |
| --- | --- | --- | --- |
| `LEWM_SOURCE_REPO` | non | Repo git cloné si `src/` est absent | `https://github.com/lucas-maes/le-wm.git` |
| `LEWM_SOURCE_REF` | non | Commit ou tag checkout après clone | `ca231f9f9d9ab041034b6d05e90b6e04bd6cff82` |

## Overrides Hydra réellement injectés

Le wrapper lance `src/train.py` avec au minimum:

```bash
python src/train.py \
  data=train__observation_images_laptop \
  subdir=<RUN_SUBDIR> \
  output_model_name=lewm \
  loader.batch_size=64 \
  loader.num_workers=16 \
  trainer.precision=bf16
```

Puis, si WandB est activé, il ajoute aussi:

```bash
wandb.enabled=true \
wandb.config.entity=<WANDB_ENTITY> \
wandb.config.project=<WANDB_PROJECT> \
wandb.config.name=<WANDB_NAME ou RUN_SUBDIR> \
wandb.config.id=<RUN_SUBDIR> \
wandb.config.resume=allow
```

## Artefacts produits

Dans `STABLEWM_HOME/<RUN_SUBDIR>/`, tu peux retrouver:
- `config.yaml`: config Hydra résolue pour le run
- `run_manifest.json`: manifeste généré par le wrapper
- `lewm_weights.ckpt`: checkpoint poids Lightning
- `lewm_epoch_<n>_object.ckpt`: checkpoints objet du modèle LeWM

Si `HF_OUTPUT_REPO_ID` est défini, ces fichiers sont uploadés sous:

```text
<HF_OUTPUT_REPO_ID>/<RUN_SUBDIR>/...
```

## Recommandations GPU cloud

- H100 / 4090: `--shm-size=16g`
- Monte `LEWM_NUM_WORKERS` à `16` ou `32` si le CPU suit
- Monte un volume persistant sur `/workspace/data` pour éviter de re-télécharger les `.h5` à chaque run
- Évite `HF_UPLOAD_ALL_CHECKPOINTS=true` sur les longs runs sauf si tu veux explicitement archiver tous les epochs

## Dépannage rapide

### Le wrapper me dit qu'il ne sait pas quel dataset utiliser

Définis explicitement:

```dotenv
LEWM_DATASET_NAME=train__observation_images_laptop
```

Ce nom doit correspondre au nom du fichier local sans extension:

```text
/workspace/data/train__observation_images_laptop.h5
```

### Le wrapper télécharge trop de fichiers

Ne laisse pas seulement `HF_DATASET_PATTERNS=*.h5` si ton repo dataset contient plusieurs `.h5` non désirés.
Utilise plutôt:

```dotenv
HF_DATASET_FILES=train/train__observation_images_laptop.h5
```

### Je veux désactiver WandB

```dotenv
WANDB_ENABLED=false
```

### Je veux forcer un run name stable

```dotenv
RUN_SUBDIR=lewm-koch-h100-test-01
WANDB_NAME=lewm-koch-h100-test-01
```
