from __future__ import annotations

import fnmatch
import inspect
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

# Headless backend for servers/containers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import stable_worldmodel as swm
except ImportError as exc:
    raise RuntimeError(
        "stable_worldmodel is required to run eval/main.py. "
        "Install eval dependencies from requirements-eval.txt."
    ) from exc

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


class IndexedSubset(Dataset):
    """Subset wrapper that keeps original dataset window indices."""

    def __init__(self, base: Dataset, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = int(self.indices[idx])
        sample = self.base[base_idx]
        sample["window_idx"] = torch.tensor(base_idx, dtype=torch.long)
        return sample


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    lowered = name.lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return requested


def _epoch_from_name(name: str) -> int:
    match = re.search(r"_epoch_(\d+)_object\.ckpt$", name)
    if match:
        return int(match.group(1))
    generic = re.search(r"_epoch_(\d+)", name)
    return int(generic.group(1)) if generic else -1


def _select_best_checkpoint(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("No checkpoint candidates available.")

    def sort_key(path: Path) -> tuple[int, float, str]:
        return (_epoch_from_name(path.name), path.stat().st_mtime, path.name)

    return sorted(paths, key=sort_key)[-1]


def find_local_checkpoint(run_id: str, local_dir: Path, pattern: str) -> Path | None:
    run_dir = local_dir / run_id
    if run_dir.exists():
        candidates = [
            p for p in run_dir.glob("**/*.ckpt") if fnmatch.fnmatch(p.name, pattern)
        ]
        if candidates:
            return _select_best_checkpoint(candidates)

    # Fallback for flat layouts where ckpts are directly in local_dir.
    direct_candidates = [
        p for p in local_dir.glob("*.ckpt") if fnmatch.fnmatch(p.name, pattern)
    ]
    if len(direct_candidates) == 1:
        return direct_candidates[0].resolve()

    # Last-resort search in local_dir when no run subfolder exists.
    if not run_dir.exists():
        recursive_candidates = [
            p for p in local_dir.glob("**/*.ckpt") if fnmatch.fnmatch(p.name, pattern)
        ]
        if len(recursive_candidates) == 1:
            return recursive_candidates[0].resolve()
    return None


def download_checkpoint_from_hf(
    repo_id: str,
    run_id: str,
    local_dir: Path,
    pattern: str,
    token: str | None,
) -> Path:
    api = HfApi(token=token)
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")

    prefix = f"{run_id}/"
    remote_candidates = [
        path
        for path in all_files
        if path.startswith(prefix) and fnmatch.fnmatch(Path(path).name, pattern)
    ]
    if not remote_candidates:
        raise RuntimeError(
            f"No remote checkpoint matched pattern {pattern!r} under {repo_id}:{run_id}/"
        )

    best_remote = sorted(
        remote_candidates,
        key=lambda x: (_epoch_from_name(Path(x).name), Path(x).name),
    )[-1]

    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=best_remote,
        local_dir=local_dir,
        token=token,
    )
    return Path(local_path).resolve()


def resolve_checkpoint(cfg: DictConfig) -> Path:
    explicit_path = cfg.checkpoint.get("path")
    if explicit_path:
        ckpt = Path(explicit_path).expanduser().resolve()
        if not ckpt.exists():
            raise RuntimeError(f"checkpoint.path does not exist: {ckpt}")
        if not ckpt.is_file():
            raise RuntimeError(f"checkpoint.path is not a file: {ckpt}")
        print(f"[checkpoint] using explicit checkpoint.path: {ckpt}")
        return ckpt

    local_dir = Path(cfg.checkpoint.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    local_ckpt = find_local_checkpoint(
        run_id=cfg.checkpoint.run_id,
        local_dir=local_dir,
        pattern=cfg.checkpoint.filename_pattern,
    )
    if local_ckpt is not None:
        print(f"[checkpoint] using local checkpoint: {local_ckpt}")
        return local_ckpt

    repo_id = cfg.checkpoint.repo_id
    if not repo_id:
        raise RuntimeError(
            "No local checkpoint found and checkpoint.repo_id is empty. "
            "Provide a HF model repo for fallback download."
        )

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    print(f"[checkpoint] local checkpoint missing, downloading from HF repo={repo_id}")
    ckpt = download_checkpoint_from_hf(
        repo_id=repo_id,
        run_id=cfg.checkpoint.run_id,
        local_dir=local_dir,
        pattern=cfg.checkpoint.filename_pattern,
        token=token,
    )
    print(f"[checkpoint] downloaded checkpoint: {ckpt}")
    return ckpt


def _load_training_cfg_for_weights(
    checkpoint_path: Path,
    cfg: DictConfig,
) -> DictConfig:
    local_candidates = [
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent.parent.parent / "config.yaml",
    ]
    for candidate in local_candidates:
        if candidate.exists() and candidate.is_file():
            print(f"[checkpoint] using training config: {candidate}")
            return OmegaConf.load(candidate)

    repo_id = cfg.checkpoint.get("repo_id")
    run_id = str(cfg.checkpoint.get("run_id") or "").strip("/")
    if not repo_id or not run_id:
        raise RuntimeError(
            "Weights checkpoint detected but no local config.yaml found. "
            "Set checkpoint.repo_id and checkpoint.run_id so eval can download config.yaml, "
            "or provide checkpoint.path to an *_object.ckpt file."
        )

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    prefix = f"{run_id}/"
    remote_cfgs = [
        path
        for path in files
        if path.startswith(prefix) and Path(path).name == "config.yaml"
    ]
    if not remote_cfgs:
        raise RuntimeError(
            f"Weights checkpoint detected but no config.yaml found under {repo_id}:{run_id}/"
        )

    remote_cfg = sorted(remote_cfgs)[-1]
    local_cfg = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=remote_cfg,
        local_dir=Path(cfg.checkpoint.local_dir).expanduser().resolve(),
        token=token,
    )
    print(f"[checkpoint] downloaded training config: {local_cfg}")
    return OmegaConf.load(local_cfg)


def _build_jepa_model_from_cfg(train_cfg: DictConfig) -> torch.nn.Module:
    try:
        import stable_pretraining as spt
        from jepa import JEPA
        from module import ARPredictor, Embedder, MLP
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import training modules needed to load weights checkpoint."
        ) from exc

    encoder = spt.backbone.utils.vit_hf(
        str(train_cfg.encoder_scale),
        patch_size=int(train_cfg.patch_size),
        image_size=int(train_cfg.img_size),
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = int(encoder.config.hidden_size)
    embed_dim = int(train_cfg.wm.get("embed_dim", hidden_dim))
    effective_act_dim = int(train_cfg.data.dataset.frameskip) * int(train_cfg.wm.action_dim)

    predictor_cfg = OmegaConf.to_container(train_cfg.predictor, resolve=True)
    if not isinstance(predictor_cfg, dict):
        raise RuntimeError("Unexpected predictor config format in training config.")

    predictor = ARPredictor(
        num_frames=int(train_cfg.wm.history_size),
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **predictor_cfg,
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

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )


def _extract_model_state_dict(checkpoint_obj: Any) -> dict[str, Any]:
    if not isinstance(checkpoint_obj, dict):
        raise RuntimeError("Unsupported checkpoint payload format for weights checkpoint.")

    state_dict = checkpoint_obj.get("state_dict")
    if state_dict is None and all(isinstance(k, str) for k in checkpoint_obj.keys()):
        state_dict = checkpoint_obj
    if not isinstance(state_dict, dict):
        raise RuntimeError("Weights checkpoint is missing a usable state_dict.")

    model_prefixed = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    if model_prefixed:
        return model_prefixed
    return state_dict


def load_dataset(cfg: DictConfig):
    dataset = swm.data.HDF5Dataset(
        name=cfg.dataset.name,
        frameskip=cfg.dataset.frameskip,
        num_steps=cfg.dataset.history_size + cfg.dataset.num_preds,
        keys_to_load=list(cfg.dataset.keys_to_load),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
        cache_dir=cfg.dataset.cache_dir,
    )
    return dataset


def compute_col_stats(dataset, col_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.asarray(dataset.get_col_data(col_name), dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]

    valid = ~np.isnan(data).any(axis=1)
    valid_data = data[valid]
    if valid_data.size == 0:
        raise RuntimeError(f"Column {col_name!r} contains no valid values for normalization.")

    mean = valid_data.mean(axis=0, keepdims=True)
    std = valid_data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.from_numpy(std.astype(np.float32)),
    )


def preprocess_pixels(pixels: torch.Tensor, img_size: int, device: torch.device) -> torch.Tensor:
    # pixels: [B, T, C, H, W], uint8
    pixels = pixels.to(device=device, dtype=torch.float32) / 255.0
    bsz, steps, channels, height, width = pixels.shape

    if height != img_size or width != img_size:
        reshaped = pixels.reshape(bsz * steps, channels, height, width)
        resized = F.interpolate(
            reshaped,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        pixels = resized.reshape(bsz, steps, channels, img_size, img_size)

    mean = IMAGENET_MEAN.to(device).view(1, 1, 3, 1, 1)
    std = IMAGENET_STD.to(device).view(1, 1, 3, 1, 1)
    return (pixels - mean) / std


def choose_windows(dataset, cfg: DictConfig) -> np.ndarray:
    rng = np.random.default_rng(int(cfg.extraction.seed))
    clip_episode_ids = np.array([ep for ep, _ in dataset.clip_indices], dtype=np.int64)
    all_episode_ids = np.unique(clip_episode_ids)

    n_req = int(cfg.extraction.num_episodes)
    if n_req > 0 and n_req < len(all_episode_ids):
        selected_episodes = np.sort(rng.choice(all_episode_ids, size=n_req, replace=False))
    else:
        selected_episodes = all_episode_ids

    mask = np.isin(clip_episode_ids, selected_episodes)
    candidate_windows = np.flatnonzero(mask)

    max_windows = int(cfg.extraction.max_windows)
    if max_windows > 0 and max_windows < len(candidate_windows):
        selected_windows = np.sort(
            rng.choice(candidate_windows, size=max_windows, replace=False)
        )
    else:
        selected_windows = candidate_windows

    print(
        f"[extract] selected {len(selected_windows)} windows "
        f"from {len(selected_episodes)} episode(s)"
    )
    return selected_windows


def _safe_tensor(batch: dict[str, Any], key: str, default: torch.Tensor) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return default
    if not torch.is_tensor(value):
        return default
    return value


def extract_latents(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
) -> dict[str, Any]:
    extract_start = time.perf_counter()
    selected_windows = choose_windows(dataset, cfg)
    subset = IndexedSubset(dataset, selected_windows)

    loader = DataLoader(
        subset,
        batch_size=int(cfg.extraction.batch_size),
        num_workers=int(cfg.extraction.num_workers),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    total_batches = len(loader)
    log_every = max(1, int(cfg.extraction.log_every_batches))
    print(
        f"[extract] dataloader batches={total_batches}, "
        f"batch_size={cfg.extraction.batch_size}, log_every={log_every}"
    )

    action_mean, action_std = compute_col_stats(dataset, "action")
    action_mean = action_mean.to(device).view(1, 1, 1, -1)
    action_std = action_std.to(device).view(1, 1, 1, -1)

    history_size = int(cfg.dataset.history_size)
    num_preds = int(cfg.dataset.num_preds)

    encode_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    mse_chunks: list[np.ndarray] = []

    episode_chunks: list[np.ndarray] = []
    step_chunks: list[np.ndarray] = []
    window_chunks: list[np.ndarray] = []
    horizon_chunks: list[np.ndarray] = []

    action_chunks: list[np.ndarray] = []
    proprio_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []

    model = model.to(device)
    model.eval()

    processed_rows = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            pixels = preprocess_pixels(
                pixels=batch["pixels"],
                img_size=int(cfg.dataset.img_size),
                device=device,
            )

            action_raw = batch["action"].to(device=device, dtype=torch.float32)
            action_dim_flat = int(action_raw.shape[-1])
            action_base_dim = int(action_mean.shape[-1])
            frame_skip = int(cfg.dataset.frameskip)

            if action_dim_flat == action_base_dim:
                action_norm = (action_raw - action_mean.view(1, 1, -1)) / action_std.view(1, 1, -1)
            elif action_dim_flat == action_base_dim * frame_skip:
                bsz, steps, _ = action_raw.shape
                action_view = action_raw.view(bsz, steps, frame_skip, action_base_dim)
                action_norm = ((action_view - action_mean) / action_std).reshape(
                    bsz, steps, action_dim_flat
                )
            else:
                raise RuntimeError(
                    "Unexpected action shape for normalization: "
                    f"flat_dim={action_dim_flat}, base_dim={action_base_dim}, "
                    f"frameskip={frame_skip}"
                )

            info = {
                "pixels": pixels,
                "action": action_norm,
            }

            output = model.encode(info)
            emb = output["emb"]
            act_emb = output["act_emb"]

            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]

            pred_emb = model.predict(ctx_emb, ctx_act)
            tgt_emb = emb[:, num_preds:]

            pred_len = min(
                pred_emb.shape[1],
                tgt_emb.shape[1],
                batch["step_idx"].shape[1] - num_preds,
            )
            if pred_len <= 0:
                continue

            pred_emb = pred_emb[:, :pred_len]
            tgt_emb = tgt_emb[:, :pred_len]
            latent_mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1)

            episode_idx = batch["episode_idx"][:, num_preds : num_preds + pred_len]
            step_idx = batch["step_idx"][:, num_preds : num_preds + pred_len]
            window_idx = batch["window_idx"].view(-1, 1).expand(-1, pred_len)
            horizon_idx = (
                torch.arange(pred_len, dtype=torch.long)
                .view(1, -1)
                .expand(window_idx.shape[0], -1)
            )

            target_action = batch["action"][:, num_preds : num_preds + pred_len]
            target_proprio = _safe_tensor(batch, "proprio", torch.empty(0))
            target_state = _safe_tensor(batch, "state", torch.empty(0))
            if target_proprio.numel() > 0:
                target_proprio = target_proprio[:, num_preds : num_preds + pred_len]
            if target_state.numel() > 0:
                target_state = target_state[:, num_preds : num_preds + pred_len]

            encode_chunks.append(tgt_emb.cpu().numpy().reshape(-1, tgt_emb.shape[-1]))
            pred_chunks.append(pred_emb.cpu().numpy().reshape(-1, pred_emb.shape[-1]))
            mse_chunks.append(latent_mse.cpu().numpy().reshape(-1))

            episode_chunks.append(episode_idx.cpu().numpy().reshape(-1))
            step_chunks.append(step_idx.cpu().numpy().reshape(-1))
            window_chunks.append(window_idx.cpu().numpy().reshape(-1))
            horizon_chunks.append(horizon_idx.cpu().numpy().reshape(-1))

            action_chunks.append(target_action.cpu().numpy().reshape(-1, target_action.shape[-1]))

            if target_proprio.numel() > 0:
                proprio_chunks.append(
                    target_proprio.cpu().numpy().reshape(-1, target_proprio.shape[-1])
                )
            if target_state.numel() > 0:
                state_chunks.append(target_state.cpu().numpy().reshape(-1, target_state.shape[-1]))

            processed_rows += int(window_idx.numel())
            if (
                batch_idx == 1
                or batch_idx % log_every == 0
                or batch_idx == total_batches
            ):
                elapsed = time.perf_counter() - extract_start
                batch_rate = batch_idx / max(elapsed, 1e-9)
                remaining_batches = total_batches - batch_idx
                eta_seconds = remaining_batches / max(batch_rate, 1e-9)
                progress = (100.0 * batch_idx) / max(total_batches, 1)
                print(
                    f"[extract] progress {batch_idx}/{total_batches} ({progress:.1f}%) "
                    f"rows={processed_rows} elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(max(0.0, eta_seconds))}"
                )

    if not encode_chunks:
        raise RuntimeError("No latent rows were extracted. Check dataset/extraction settings.")

    encode_latents = np.concatenate(encode_chunks, axis=0)
    pred_latents = np.concatenate(pred_chunks, axis=0)
    latent_mse = np.concatenate(mse_chunks, axis=0)

    episode_idx = np.concatenate(episode_chunks, axis=0).astype(np.int64)
    step_idx = np.concatenate(step_chunks, axis=0).astype(np.int64)
    window_idx = np.concatenate(window_chunks, axis=0).astype(np.int64)
    horizon_idx = np.concatenate(horizon_chunks, axis=0).astype(np.int64)

    targets: dict[str, np.ndarray] = {
        "action": np.concatenate(action_chunks, axis=0).astype(np.float32),
    }
    if proprio_chunks:
        targets["proprio"] = np.concatenate(proprio_chunks, axis=0).astype(np.float32)
    if state_chunks:
        targets["state"] = np.concatenate(state_chunks, axis=0).astype(np.float32)

    metadata = pd.DataFrame(
        {
            "row_id": np.arange(encode_latents.shape[0], dtype=np.int64),
            "window_idx": window_idx,
            "horizon_idx": horizon_idx,
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "latent_mse": latent_mse,
        }
    )

    print(
        f"[extract] completed with {processed_rows} latent rows "
        f"(embedding_dim={encode_latents.shape[1]}) in "
        f"{format_duration(time.perf_counter() - extract_start)}"
    )

    return {
        "encode_latents": encode_latents,
        "pred_latents": pred_latents,
        "metadata": metadata,
        "targets": targets,
    }


def run_pca(
    encode_latents: np.ndarray,
    pred_latents: np.ndarray,
    metadata: pd.DataFrame,
    output_csv: Path,
) -> dict[str, Any]:
    x = np.concatenate([encode_latents, pred_latents], axis=0)
    pca = PCA(n_components=2)
    pca.fit(x)

    encode_2d = pca.transform(encode_latents)
    pred_2d = pca.transform(pred_latents)

    pca_df = metadata.copy()
    pca_df["encode_pca_x"] = encode_2d[:, 0]
    pca_df["encode_pca_y"] = encode_2d[:, 1]
    pca_df["pred_pca_x"] = pred_2d[:, 0]
    pca_df["pred_pca_y"] = pred_2d[:, 1]
    pca_df.to_csv(output_csv, index=False)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "csv": str(output_csv),
        "dataframe": pca_df,
    }


def build_tsne_model(cfg: DictConfig, n_samples: int) -> TSNE:
    perplexity = float(cfg.projection.tsne.perplexity)
    max_perplexity = max(1.0, (n_samples - 1) / 3.0)
    perplexity = min(perplexity, max_perplexity)

    kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": float(cfg.projection.tsne.learning_rate),
        "early_exaggeration": float(cfg.projection.tsne.early_exaggeration),
        "init": str(cfg.projection.tsne.init),
        "random_state": int(cfg.extraction.seed),
    }

    tsne_sig = inspect.signature(TSNE.__init__)
    if "max_iter" in tsne_sig.parameters:
        kwargs["max_iter"] = int(cfg.projection.tsne.n_iter)
    else:
        kwargs["n_iter"] = int(cfg.projection.tsne.n_iter)

    return TSNE(**kwargs)


def run_tsne(
    cfg: DictConfig,
    encode_latents: np.ndarray,
    pred_latents: np.ndarray,
    metadata: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    max_points = int(cfg.projection.tsne.max_points)
    n_rows = metadata.shape[0]
    if n_rows < 5:
        empty = pd.DataFrame(
            columns=[
                "row_id",
                "window_idx",
                "horizon_idx",
                "episode_idx",
                "step_idx",
                "point_type",
                "tsne_x",
                "tsne_y",
            ]
        )
        empty.to_csv(output_csv, index=False)
        return empty

    rng = np.random.default_rng(int(cfg.extraction.seed))
    sample_size = min(max_points, n_rows)
    sampled_rows = np.sort(rng.choice(n_rows, size=sample_size, replace=False))

    x = np.concatenate(
        [encode_latents[sampled_rows], pred_latents[sampled_rows]],
        axis=0,
    )
    print(
        f"[tsne] fitting on {x.shape[0]} points "
        f"(sampled_rows={sample_size}, n_iter={cfg.projection.tsne.n_iter}, "
        f"seed={cfg.extraction.seed})"
    )
    model = build_tsne_model(cfg, x.shape[0])
    tsne_2d = model.fit_transform(x)

    sampled_meta = metadata.iloc[sampled_rows].reset_index(drop=True)
    encode_df = sampled_meta.copy()
    pred_df = sampled_meta.copy()

    encode_df["point_type"] = "encode"
    pred_df["point_type"] = "pred"

    encode_df["tsne_x"] = tsne_2d[:sample_size, 0]
    encode_df["tsne_y"] = tsne_2d[:sample_size, 1]

    pred_df["tsne_x"] = tsne_2d[sample_size:, 0]
    pred_df["tsne_y"] = tsne_2d[sample_size:, 1]

    tsne_df = pd.concat([encode_df, pred_df], ignore_index=True)
    tsne_df.to_csv(output_csv, index=False)
    print(f"[tsne] wrote {output_csv} with {len(tsne_df)} rows")
    return tsne_df


def run_neighbors(
    cfg: DictConfig,
    encode_latents: np.ndarray,
    metadata: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    n_samples = encode_latents.shape[0]
    if n_samples < 2:
        empty = pd.DataFrame(
            columns=[
                "anchor_row_id",
                "neighbor_row_id",
                "rank",
                "distance",
                "anchor_episode_idx",
                "neighbor_episode_idx",
                "anchor_step_idx",
                "neighbor_step_idx",
            ]
        )
        empty.to_csv(output_csv, index=False)
        return empty

    rng = np.random.default_rng(int(cfg.extraction.seed))
    n_anchors = min(int(cfg.neighbors.num_anchors), n_samples)
    k = min(int(cfg.neighbors.k), n_samples - 1)

    anchor_ids = np.sort(rng.choice(n_samples, size=n_anchors, replace=False))
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_samples), metric="euclidean")
    nn.fit(encode_latents)
    distances, neighbors = nn.kneighbors(encode_latents[anchor_ids], return_distance=True)

    rows: list[dict[str, Any]] = []
    for anchor_local_idx, anchor_row in enumerate(anchor_ids):
        anchor_neighbors = neighbors[anchor_local_idx]
        anchor_dist = distances[anchor_local_idx]

        rank = 0
        for cand_idx, dist in zip(anchor_neighbors, anchor_dist):
            if cand_idx == anchor_row:
                continue
            rank += 1
            rows.append(
                {
                    "anchor_row_id": int(metadata.iloc[anchor_row]["row_id"]),
                    "neighbor_row_id": int(metadata.iloc[cand_idx]["row_id"]),
                    "rank": rank,
                    "distance": float(dist),
                    "anchor_episode_idx": int(metadata.iloc[anchor_row]["episode_idx"]),
                    "neighbor_episode_idx": int(metadata.iloc[cand_idx]["episode_idx"]),
                    "anchor_step_idx": int(metadata.iloc[anchor_row]["step_idx"]),
                    "neighbor_step_idx": int(metadata.iloc[cand_idx]["step_idx"]),
                }
            )
            if rank >= k:
                break

    nn_df = pd.DataFrame(rows)
    nn_df.to_csv(output_csv, index=False)
    return nn_df


def probe_regression(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    if features.shape[0] != targets.shape[0]:
        raise ValueError("Feature and target lengths mismatch for probing.")

    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 2:
        n_samples = features.shape[0]
        split_at = max(1, int((1.0 - test_size) * n_samples))
        train_idx = np.arange(0, split_at)
        test_idx = np.arange(split_at, n_samples)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(features, targets, groups=groups))

    x_train = features[train_idx]
    x_test = features[test_idx]
    y_train = targets[train_idx]
    y_test = targets[test_idx]

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = float(r2_score(y_test, y_pred, multioutput="uniform_average"))
    mse = float(mean_squared_error(y_test, y_pred, multioutput="uniform_average"))

    r2_per_dim = r2_score(y_test, y_pred, multioutput="raw_values")
    mse_per_dim = mean_squared_error(y_test, y_pred, multioutput="raw_values")

    return {
        "r2": r2,
        "mse": mse,
        "r2_per_dim": np.asarray(r2_per_dim, dtype=np.float64).tolist(),
        "mse_per_dim": np.asarray(mse_per_dim, dtype=np.float64).tolist(),
        "n_train": int(train_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
    }


def run_probes(cfg: DictConfig, bundle: dict[str, Any], output_json: Path) -> dict[str, Any]:
    encode = bundle["encode_latents"]
    pred = bundle["pred_latents"]
    groups = bundle["metadata"]["episode_idx"].to_numpy()

    results: dict[str, Any] = {
        "settings": {
            "test_size": float(cfg.probe.test_size),
            "ridge_alpha": float(cfg.probe.ridge_alpha),
            "seed": int(cfg.extraction.seed),
        },
        "encode": {},
        "pred": {},
    }

    for signal_name, targets in bundle["targets"].items():
        print(f"[probe] fitting linear probes for signal={signal_name}")
        results["encode"][signal_name] = probe_regression(
            features=encode,
            targets=targets,
            groups=groups,
            test_size=float(cfg.probe.test_size),
            alpha=float(cfg.probe.ridge_alpha),
            seed=int(cfg.extraction.seed),
        )
        results["pred"][signal_name] = probe_regression(
            features=pred,
            targets=targets,
            groups=groups,
            test_size=float(cfg.probe.test_size),
            alpha=float(cfg.probe.ridge_alpha),
            seed=int(cfg.extraction.seed),
        )

    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[probe] wrote {output_json}")
    return results


def plot_pca_scatter(pca_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Episode-coded colors for readability.
    colors = pca_df["episode_idx"].to_numpy()

    axes[0].scatter(
        pca_df["encode_pca_x"],
        pca_df["encode_pca_y"],
        c=colors,
        s=6,
        cmap="viridis",
        alpha=0.7,
    )
    axes[0].set_title("PCA Encode Latents")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(
        pca_df["pred_pca_x"],
        pca_df["pred_pca_y"],
        c=colors,
        s=6,
        cmap="viridis",
        alpha=0.7,
    )
    axes[1].set_title("PCA Predicted Latents")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_tsne_scatter(tsne_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if tsne_df.empty:
        ax.text(0.5, 0.5, "Not enough points for t-SNE", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return

    encode_df = tsne_df[tsne_df["point_type"] == "encode"]
    pred_df = tsne_df[tsne_df["point_type"] == "pred"]

    ax.scatter(
        encode_df["tsne_x"],
        encode_df["tsne_y"],
        s=8,
        alpha=0.7,
        label="encode",
    )
    ax.scatter(
        pred_df["tsne_x"],
        pred_df["tsne_y"],
        s=8,
        alpha=0.7,
        label="pred",
    )

    ax.set_title("t-SNE Latents")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_trajectories(
    cfg: DictConfig,
    pca_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    episodes = np.sort(pca_df["episode_idx"].unique())
    if episodes.size == 0:
        return

    max_ep = int(cfg.viz.trajectories.num_episodes_plot)
    if max_ep > 0 and episodes.size > max_ep:
        rng = np.random.default_rng(int(cfg.extraction.seed))
        episodes = np.sort(rng.choice(episodes, size=max_ep, replace=False))

    n = len(episodes)
    n_cols = int(math.ceil(math.sqrt(n)))
    n_rows = int(math.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        constrained_layout=True,
    )

    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax in axes_arr[n:]:
        ax.set_axis_off()

    for ax, ep in zip(axes_arr, episodes):
        episode_df = pca_df[pca_df["episode_idx"] == ep].copy()
        step_agg = (
            episode_df.groupby("step_idx", as_index=False)[
                [
                    "encode_pca_x",
                    "encode_pca_y",
                    "pred_pca_x",
                    "pred_pca_y",
                    "latent_mse",
                ]
            ]
            .mean()
            .sort_values("step_idx")
        )

        ax.plot(step_agg["encode_pca_x"], step_agg["encode_pca_y"], label="encode")
        ax.plot(
            step_agg["pred_pca_x"],
            step_agg["pred_pca_y"],
            linestyle="--",
            label="pred",
        )
        ax.set_title(f"Episode {int(ep)}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_mse_curves(pca_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    grouped = pca_df.groupby("step_idx", as_index=False)["latent_mse"].agg(["mean", "std"])
    grouped = grouped.reset_index()

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = grouped["step_idx"].to_numpy()
    mean = grouped["mean"].to_numpy()
    std = grouped["std"].fillna(0.0).to_numpy()

    ax.plot(x, mean, label="Mean latent MSE")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std")
    ax.set_xlabel("step_idx")
    ax.set_ylabel("latent_mse")
    ax.set_title("Latent Prediction Error by Step")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_neighbors(
    pca_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    ax.scatter(
        pca_df["encode_pca_x"],
        pca_df["encode_pca_y"],
        c="lightgray",
        s=6,
        alpha=0.5,
        label="all encode points",
    )

    if not nn_df.empty:
        row_to_xy = pca_df.set_index("row_id")[["encode_pca_x", "encode_pca_y"]]

        anchor_ids = nn_df["anchor_row_id"].unique()
        anchor_xy = row_to_xy.loc[anchor_ids]
        ax.scatter(
            anchor_xy["encode_pca_x"],
            anchor_xy["encode_pca_y"],
            c="tab:red",
            s=40,
            marker="x",
            label="anchors",
        )

        neigh_ids = nn_df["neighbor_row_id"].unique()
        neigh_xy = row_to_xy.loc[neigh_ids]
        ax.scatter(
            neigh_xy["encode_pca_x"],
            neigh_xy["encode_pca_y"],
            c="tab:blue",
            s=20,
            alpha=0.8,
            label="neighbors",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Nearest Neighbors in PCA Space")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_probe_metrics(probe_metrics: dict[str, Any], out_path: Path, dpi: int) -> None:
    signals = sorted(probe_metrics["encode"].keys())
    if not signals:
        return

    encode_r2 = [probe_metrics["encode"][s]["r2"] for s in signals]
    pred_r2 = [probe_metrics["pred"][s]["r2"] for s in signals]

    x = np.arange(len(signals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width / 2, encode_r2, width=width, label="encode")
    ax.bar(x + width / 2, pred_r2, width=width, label="pred")

    ax.set_xticks(x)
    ax.set_xticklabels(signals)
    ax.set_ylabel("R2")
    ax.set_title("Linear Probe R2 by Signal")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_report(
    cfg: DictConfig,
    checkpoint_path: Path,
    bundle: dict[str, Any],
    pca_stats: dict[str, Any],
    probe_metrics: dict[str, Any],
    artifact_dir: Path,
) -> None:
    metadata = bundle["metadata"]
    n_rows = int(metadata.shape[0])
    n_windows = int(metadata["window_idx"].nunique())
    n_episodes = int(metadata["episode_idx"].nunique())

    report_path = artifact_dir / "report.md"

    lines: list[str] = []
    lines.append("# JEPA KOCH Latent Evaluation Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- run_id: `{cfg.output.run_id}`")
    lines.append(f"- checkpoint: `{checkpoint_path}`")
    lines.append(f"- dataset: `{cfg.dataset.name}`")
    lines.append(f"- rows: `{n_rows}`")
    lines.append(f"- windows: `{n_windows}`")
    lines.append(f"- episodes: `{n_episodes}`")
    lines.append("")

    lines.append("## PCA")
    explained = pca_stats["explained_variance_ratio"]
    lines.append(
        "- explained_variance_ratio (PC1, PC2): "
        f"`[{explained[0]:.4f}, {explained[1]:.4f}]`"
    )
    lines.append("")

    lines.append("## Probing Metrics")
    for latent_type in ("encode", "pred"):
        lines.append(f"### {latent_type}")
        for signal, metrics in probe_metrics[latent_type].items():
            lines.append(
                f"- {signal}: R2={metrics['r2']:.4f}, MSE={metrics['mse']:.6f}, "
                f"n_train={metrics['n_train']}, n_test={metrics['n_test']}"
            )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("- `latent_pca.csv`")
    lines.append("- `latent_tsne.csv`")
    lines.append("- `probe_metrics.json`")
    lines.append("- `nearest_neighbors.csv`")
    lines.append("- `figures/*.png`")
    lines.append("")

    lines.append("## Interpretation Caveats")
    lines.append("- 2D projections can hide high-dimensional geometry.")
    lines.append("- t-SNE neighborhoods depend on perplexity, seed and sample selection.")
    lines.append("- Visual cluster proximity does not imply causal robot behavior.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_output_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    artifact_dir = Path(cfg.output.artifact_root).expanduser().resolve() / cfg.output.run_id
    figures_dir = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir, figures_dir


def save_optional_extraction_table(cfg: DictConfig, bundle: dict[str, Any], artifact_dir: Path) -> None:
    if not bool(cfg.output.save_extraction_table):
        return

    metadata = bundle["metadata"].copy()
    action = bundle["targets"]["action"]
    for i in range(action.shape[1]):
        metadata[f"action_{i}"] = action[:, i]

    for key in ("proprio", "state"):
        if key in bundle["targets"]:
            values = bundle["targets"][key]
            for i in range(values.shape[1]):
                metadata[f"{key}_{i}"] = values[:, i]

    metadata.to_csv(artifact_dir / "latent_extraction.csv", index=False)


def load_checkpoint_model(path: Path, device: torch.device, cfg: DictConfig) -> torch.nn.Module:
    # PyTorch >=2.6 defaults to weights_only=True, but LeWM object checkpoints
    # contain serialized module objects.
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)

    if isinstance(payload, torch.nn.Module):
        return payload

    # Fallback for lightning/state_dict checkpoints (e.g. *_weights.ckpt).
    if path.name.endswith("_weights.ckpt"):
        train_cfg = _load_training_cfg_for_weights(path, cfg)
        model = _build_jepa_model_from_cfg(train_cfg)
        state_dict = _extract_model_state_dict(payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"Weights checkpoint load failed: missing {len(missing)} keys "
                f"(sample: {missing[:5]})"
            )
        if unexpected:
            print(
                f"[checkpoint] warning: ignored {len(unexpected)} unexpected weights "
                f"(sample: {unexpected[:5]})"
            )
        return model.to(device)

    raise RuntimeError(
        f"Checkpoint did not contain a torch.nn.Module: {path}. "
        "Use an *_object.ckpt or a compatible *_weights.ckpt."
    )


def validate_contract(artifact_dir: Path) -> None:
    required = [
        artifact_dir / "latent_pca.csv",
        artifact_dir / "latent_tsne.csv",
        artifact_dir / "probe_metrics.json",
        artifact_dir / "figures",
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Missing required artifact: {path}")

    pca_df = pd.read_csv(artifact_dir / "latent_pca.csv")
    required_cols = {"episode_idx", "step_idx", "encode_pca_x", "encode_pca_y"}
    missing = required_cols - set(pca_df.columns)
    if missing:
        raise RuntimeError(f"latent_pca.csv missing required columns: {sorted(missing)}")


def print_config(cfg: DictConfig) -> None:
    print("[config] --------")
    print(OmegaConf.to_yaml(cfg))
    print("[config] --------")


@hydra.main(version_base=None, config_path="config", config_name="eval")
def run(cfg: DictConfig) -> None:
    run_start = time.perf_counter()
    set_global_seeds(int(cfg.extraction.seed))
    print_config(cfg)

    device = resolve_device(str(cfg.extraction.device))
    print(f"[device] using {device}")

    t0 = time.perf_counter()
    print("[stage] resolving checkpoint")
    checkpoint_path = resolve_checkpoint(cfg)
    print(f"[stage] resolve checkpoint done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] loading checkpoint model")
    model = load_checkpoint_model(checkpoint_path, device, cfg)
    print(f"[stage] load model done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] loading dataset")
    dataset = load_dataset(cfg)
    print(
        f"[stage] dataset loaded in {format_duration(time.perf_counter() - t0)} "
        f"(windows={len(dataset)}, columns={dataset.column_names})"
    )

    t0 = time.perf_counter()
    print("[stage] extracting latents")
    bundle = extract_latents(cfg, model, dataset, device)
    print(f"[stage] extract latents done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] preparing output directories")
    artifact_dir, figures_dir = ensure_output_dirs(cfg)
    print(
        f"[stage] output directories ready in {format_duration(time.perf_counter() - t0)} "
        f"({artifact_dir})"
    )

    t0 = time.perf_counter()
    print("[stage] PCA projection")
    pca_stats = run_pca(
        encode_latents=bundle["encode_latents"],
        pred_latents=bundle["pred_latents"],
        metadata=bundle["metadata"],
        output_csv=artifact_dir / "latent_pca.csv",
    )
    print(f"[stage] PCA done in {format_duration(time.perf_counter() - t0)}")
    pca_df = pca_stats["dataframe"]

    t0 = time.perf_counter()
    print("[stage] t-SNE projection")
    tsne_df = run_tsne(
        cfg=cfg,
        encode_latents=bundle["encode_latents"],
        pred_latents=bundle["pred_latents"],
        metadata=bundle["metadata"],
        output_csv=artifact_dir / "latent_tsne.csv",
    )
    print(f"[stage] t-SNE done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] nearest neighbors")
    nn_df = run_neighbors(
        cfg=cfg,
        encode_latents=bundle["encode_latents"],
        metadata=bundle["metadata"],
        output_csv=artifact_dir / "nearest_neighbors.csv",
    )
    print(
        f"[stage] nearest neighbors done in {format_duration(time.perf_counter() - t0)} "
        f"(rows={len(nn_df)})"
    )

    t0 = time.perf_counter()
    print("[stage] probing")
    probe_metrics = run_probes(
        cfg=cfg,
        bundle=bundle,
        output_json=artifact_dir / "probe_metrics.json",
    )
    print(f"[stage] probing done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] rendering figures")
    plot_pca_scatter(
        pca_df=pca_df,
        out_path=figures_dir / "pca_scatter.png",
        dpi=int(cfg.viz.dpi),
    )
    plot_tsne_scatter(
        tsne_df=tsne_df,
        out_path=figures_dir / "tsne_scatter.png",
        dpi=int(cfg.viz.dpi),
    )
    plot_trajectories(
        cfg=cfg,
        pca_df=pca_df,
        out_path=figures_dir / "trajectories_pca.png",
        dpi=int(cfg.viz.dpi),
    )
    plot_mse_curves(
        pca_df=pca_df,
        out_path=figures_dir / "latent_mse_by_step.png",
        dpi=int(cfg.viz.dpi),
    )
    plot_neighbors(
        pca_df=pca_df,
        nn_df=nn_df,
        out_path=figures_dir / "nearest_neighbors_pca.png",
        dpi=int(cfg.viz.dpi),
    )
    plot_probe_metrics(
        probe_metrics=probe_metrics,
        out_path=figures_dir / "probe_r2.png",
        dpi=int(cfg.viz.dpi),
    )
    print(f"[stage] figures done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] writing optional extraction table")
    save_optional_extraction_table(cfg, bundle, artifact_dir)
    print(f"[stage] optional table done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] writing report")
    write_report(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        bundle=bundle,
        pca_stats=pca_stats,
        probe_metrics=probe_metrics,
        artifact_dir=artifact_dir,
    )
    print(f"[stage] report done in {format_duration(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    print("[stage] validating artifact contract")
    validate_contract(artifact_dir)
    print(f"[stage] validation done in {format_duration(time.perf_counter() - t0)}")

    print(
        f"[done] artifacts written to {artifact_dir} "
        f"in {format_duration(time.perf_counter() - run_start)}"
    )


if __name__ == "__main__":
    run()
