import os
from functools import partial
from pathlib import Path

import sitecustomize  # noqa: F401
import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import ConcatDataset, Dataset
from lightning.pytorch.loggers import WandbLogger
from omegaconf import ListConfig, OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import HFCheckpointSyncCallback, ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


class EpisodeShuffledConcatDataset(Dataset):
    """Concatenate datasets and shuffle at episode granularity."""

    def __init__(self, datasets: list[swm.data.HDF5Dataset], seed: int):
        super().__init__()
        self.concat = ConcatDataset(datasets)
        episode_groups = []
        offset = 0
        for dataset in datasets:
            grouped = {}
            for local_idx, (episode_idx, _) in enumerate(dataset.clip_indices):
                grouped.setdefault(int(episode_idx), []).append(offset + local_idx)
            episode_groups.extend(grouped.values())
            offset += len(dataset)

        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(episode_groups), generator=generator).tolist()
        self.indices = [sample_idx for group_idx in permutation for sample_idx in episode_groups[group_idx]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.concat[self.indices[idx]]


def _dataset_names_from_cfg(name_field) -> list[str]:
    if isinstance(name_field, ListConfig):
        return [str(item) for item in name_field]
    if isinstance(name_field, (list, tuple)):
        return [str(item) for item in name_field]
    return [str(name_field)]


def _check_dataset_compatibility(datasets: list[swm.data.HDF5Dataset], keys_to_load: list[str]) -> None:
    if not datasets:
        raise ValueError('No dataset provided.')
    reference_keys = set(datasets[0].column_names)
    for idx, dataset in enumerate(datasets[1:], start=1):
        current_keys = set(dataset.column_names)
        if current_keys != reference_keys:
            raise ValueError(f'Dataset schema mismatch at index {idx}: {current_keys} vs {reference_keys}')
    for key in keys_to_load:
        if key.startswith('pixels'):
            continue
        reference_dim = datasets[0].get_dim(key)
        for idx, dataset in enumerate(datasets[1:], start=1):
            if dataset.get_dim(key) != reference_dim:
                raise ValueError(
                    f'Dataset dim mismatch for key={key!r}: dataset 0 has {reference_dim}, dataset {idx} has {dataset.get_dim(key)}'
                )


def _get_column_normalizer_for_datasets(
    datasets: list[swm.data.HDF5Dataset],
    source: str,
    target: str,
):
    if len(datasets) == 1:
        return get_column_normalizer(datasets[0], source, target)

    merged = np.concatenate([dataset.get_col_data(source) for dataset in datasets], axis=0)
    data = torch.from_numpy(np.array(merged))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    return spt.data.transforms.WrapTorchTransform(norm_fn, source=source, target=target)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = dict(cfg.data.dataset)
    dataset_names = _dataset_names_from_cfg(dataset_cfg.pop('name'))
    datasets = [
        swm.data.HDF5Dataset(name=dataset_name, transform=None, **dataset_cfg)
        for dataset_name in dataset_names
    ]
    _check_dataset_compatibility(datasets, list(cfg.data.dataset.keys_to_load))
    dataset = datasets[0] if len(datasets) == 1 else EpisodeShuffledConcatDataset(datasets, seed=cfg.seed)
    print(f'[data] training datasets: {dataset_names}', flush=True)

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = _get_column_normalizer_for_datasets(datasets, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", datasets[0].get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    if len(datasets) == 1:
        dataset.transform = transform
    else:
        # In merged mode, apply transforms inside each HDF5Dataset _before_
        # the base Dataset reshapes action to (num_steps, frameskip * action_dim).
        for base_dataset in datasets:
            base_dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=int(cfg.checkpoint.object_epoch_interval),
    )
    callbacks = [object_dump_callback]

    push_on_eval = os.getenv('LEWM_PUSH_CHECKPOINT_ON_EVAL', 'true').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    output_repo_id = os.getenv('HF_OUTPUT_REPO_ID')
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if push_on_eval and output_repo_id and hf_token:
        callbacks.append(
            HFCheckpointSyncCallback(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                run_subdir=run_id,
                repo_id=output_repo_id,
                token=hf_token,
                private_repo=os.getenv('HF_OUTPUT_PRIVATE', 'true').strip().lower() in {'1', 'true', 'yes', 'y', 'on'},
                layout=os.getenv('LEWM_HF_CHECKPOINT_LAYOUT', 'transformers'),
                push_on_eval=True,
            )
        )
    elif push_on_eval:
        print('[sync-eval] disabled: set HF_OUTPUT_REPO_ID and HF_TOKEN to enable upload on eval.', flush=True)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
