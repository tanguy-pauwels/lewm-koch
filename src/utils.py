import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


class HFCheckpointSyncCallback(Callback):
    """Upload checkpoints to HF Hub after validation epochs and at training end."""

    def __init__(
        self,
        dirpath,
        filename,
        run_subdir,
        repo_id,
        token,
        *,
        private_repo=True,
        layout='transformers',
        push_on_eval=True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.run_subdir = run_subdir
        self.repo_id = repo_id
        self.token = token
        self.private_repo = private_repo
        self.layout = (layout or 'transformers').strip().lower()
        self.push_on_eval = push_on_eval
        self._api = None

    def _ensure_api(self):
        if self._api is not None:
            return self._api
        if HfApi is None:
            raise RuntimeError('huggingface_hub is not available in the training environment.')
        self._api = HfApi(token=self.token)
        self._api.create_repo(
            repo_id=self.repo_id,
            repo_type='model',
            private=self.private_repo,
            exist_ok=True,
        )
        return self._api

    def _folder_for_checkpoint(self, checkpoint_name):
        if self.layout == 'transformers':
            return f'{self.run_subdir}/checkpoints/{checkpoint_name}'
        if self.layout in {'epoch', 'epochs'}:
            return f'{self.run_subdir}/epochs/{checkpoint_name}'
        return self.run_subdir

    def _collect_files(self, epoch):
        files = [
            self.dirpath / 'config.yaml',
            self.dirpath / 'run_manifest.json',
            self.dirpath / f'{self.filename}_weights.ckpt',
        ]
        if epoch is not None:
            files.append(self.dirpath / f'{self.filename}_epoch_{epoch}_object.ckpt')
        else:
            object_ckpts = sorted(self.dirpath.glob(f'{self.filename}_epoch_*_object.ckpt'))
            if object_ckpts:
                files.append(object_ckpts[-1])
        return [path for path in files if path.exists()]

    def _upload(self, checkpoint_name, epoch):
        files = self._collect_files(epoch=epoch)
        if not files:
            print(f'[sync-eval] no checkpoint files found for {checkpoint_name}', flush=True)
            return
        api = self._ensure_api()
        folder = self._folder_for_checkpoint(checkpoint_name)
        for local_file in files:
            path_in_repo = f'{folder}/{local_file.name}'
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type='model',
            )
        print(f'[sync-eval] uploaded {len(files)} file(s) to {self.repo_id}:{folder}', flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if trainer.sanity_checking or not trainer.is_global_zero or not self.push_on_eval:
            return
        epoch = trainer.current_epoch + 1
        checkpoint_name = f'checkpoint-{epoch:05d}'
        try:
            self._upload(checkpoint_name=checkpoint_name, epoch=epoch)
        except Exception as e:
            print(f'[sync-eval] upload failed for epoch {epoch}: {e}', flush=True)

    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        try:
            self._upload(checkpoint_name='checkpoint-final', epoch=None)
        except Exception as e:
            print(f'[sync-eval] final upload failed: {e}', flush=True)
