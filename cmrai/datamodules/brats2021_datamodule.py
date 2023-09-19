import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import wandb
from cmrai import utils

# for debugging
# from albumentations import ReplayCompose as Compose

log = utils.get_pylogger(__name__)
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


class LoaderWrapper:
    def __init__(self, dataloader, n_step):
        self.step = n_step
        self.idx = 0
        self.dataloader = dataloader
        self.iter_loader = iter(self.dataloader)

    def __iter__(self):
        return self

    def __len__(self):
        return self.step

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)


class BRATS2021DataModule(LightningDataModule):
    """_summary_

    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

    Args:
        LightningDataModule (_type_): _description_
    """

    def __init__(
        self,
        data_transform: DictConfig,
        data_dir: str,
        processed_artefact: str,
        train_artefact: str,
        val_artefact: str,
        test_artefact: str,
        n_step_per_epoch: int,
        batch_size: int,
        image_dim: int,
        data_mean: float,
        data_std: float,
        max_pixel_value: float,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        prefetch_factor: int = None,
        unpaired_data: bool = False,
        shuffle_training: bool = True,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if self.hparams.image_dim == 2:
            from albumentations import Compose
        elif self.hparams.image_dim == 3:
            from monai.transforms import Compose
        else:
            raise ValueError("image_dim must be 2 or 3")

        # data transformations
        transforms_start = []
        self.transforms_train = []
        self.transforms_target = []
        self.transforms_val = []
        self.transforms_final = []
        if data_transform.get("start"):
            for _, da_conf in data_transform.start.items():
                if "_target_" in da_conf:
                    log.info(
                        f"Instantiating starting data transformation <{da_conf._target_}>"
                    )
                    transforms_start.append(
                        hydra.utils.instantiate(da_conf, _convert_="all")
                    )

        if data_transform.get("train"):
            for _, da_conf in data_transform.train.items():
                if "_target_" in da_conf:
                    log.info(
                        f"Instantiating training data transformation <{da_conf._target_}>"
                    )
                    self.transforms_train.append(
                        hydra.utils.instantiate(da_conf, _convert_="all")
                    )

        if data_transform.get("target"):
            for _, da_conf in data_transform.target.items():
                if "_target_" in da_conf:
                    log.info(
                        f"Instantiating target data transformation of training data <{da_conf._target_}>"
                    )
                    self.transforms_target.append(
                        hydra.utils.instantiate(da_conf, _convert_="all")
                    )

        if data_transform.get("val"):
            for _, da_conf in data_transform.val.items():
                if "_target_" in da_conf:
                    log.info(
                        f"Instantiating validation data transformation <{da_conf._target_}>"
                    )
                    self.transforms_val.append(
                        hydra.utils.instantiate(da_conf, _convert_="all")
                    )

        if data_transform.get("final"):
            for _, da_conf in data_transform.final.items():
                if "_target_" in da_conf:
                    log.info(
                        f"Instantiating final data transformation for all <{da_conf._target_}>"
                    )
                    transformation = hydra.utils.instantiate(da_conf, _convert_="all")
                    self.transforms_final.append(transformation)

        if self.hparams.image_dim == 2:
            self.transforms_train = Compose(
                transforms_start + self.transforms_train,
                additional_targets={"target": "image"},
            )
            self.transforms_target = Compose(self.transforms_target)
            self.transforms_val = Compose(
                transforms_start + self.transforms_val,
                additional_targets={"target": "image"},
            )
            self.transforms_final = Compose(
                self.transforms_final, additional_targets={"target": "image"}
            )
        elif self.hparams.image_dim == 3:
            self.transforms_train = Compose(transforms_start + self.transforms_train)
            self.transforms_target = Compose(self.transforms_target)
            self.transforms_val = Compose(transforms_start + self.transforms_val)
            self.transforms_final = Compose(self.transforms_final)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.processed_path: Optional[Path] = None
        self.train_path: Optional[Path] = None
        self.val_path: Optional[Path] = None
        self.test_path: Optional[Path] = None

        self.wandb_prepare_data()

    def _set_data_path(self):
        api = wandb.Api()
        artifact = api.artifact(
            name=f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/"
            + self.hparams.processed_artefact,
            type="processed_data",
        )
        file_name, data_path = self._get_path_from_artifact(artifact)
        setattr(self, "processed_path", data_path / file_name)

        if self.hparams.train_artefact:
            train_artifact = api.artifact(
                name=f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/"
                + self.hparams.train_artefact,
                type="train_data",
            )
            file_name, data_path = self._get_path_from_artifact(train_artifact)
            setattr(self, "train_path", data_path / file_name)

        if self.hparams.val_artefact:
            val_artifact = api.artifact(
                name=f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/"
                + self.hparams.val_artefact,
                type="val_data",
            )
            file_name, data_path = self._get_path_from_artifact(val_artifact)
            setattr(self, "val_path", data_path / file_name)

        if self.hparams.test_artefact:
            test_artifact = api.artifact(
                name=f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/"
                + self.hparams.test_artefact,
                type="test_data",
            )
            file_name, data_path = self._get_path_from_artifact(test_artifact)
            setattr(self, "test_path", data_path / file_name)

    def _get_path_from_artifact(self, artifact):
        file_name, _ = next(iter(artifact.manifest.entries.items()))
        try:
            data_path = root / artifact.metadata["subfolder"]
        except KeyError:
            data_path = root / self.hparams.data_dir / "BRATS2021/processed"
        return file_name, data_path

    def _download_artifact(self, data_artifact, attr_name):
        """Download data artifact from W&B and save it to self."""
        file_name, data_path = self._get_path_from_artifact(data_artifact)
        data_path.mkdir(parents=True, exist_ok=True)

        # Download data
        log.info(
            f"Downloading {file_name} with version id[{data_artifact.name.split(':')[1]}] from artefact: {data_artifact.name.split(':')[0]}..."
        )
        (data_path / file_name).unlink(missing_ok=True)
        data_artifact.download(root=str(data_path))

    def wandb_prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # Setup WandB artifacts
        if wandb.run:
            data_artifact = wandb.run.use_artifact(
                self.hparams.processed_artefact, type="processed_data"
            )
            self._download_artifact(data_artifact, "processed_path")

            if self.hparams.train_artefact:
                train_artifact = wandb.run.use_artifact(
                    self.hparams.train_artefact, type="train_data"
                )
                self._download_artifact(train_artifact, "train_path")

            if self.hparams.val_artefact:
                val_artifact = wandb.run.use_artifact(
                    self.hparams.val_artefact, type="val_data"
                )
                self._download_artifact(val_artifact, "val_path")

            if self.hparams.test_artefact:
                test_artifact = wandb.run.use_artifact(
                    self.hparams.test_artefact, type="test_data"
                )
                self._download_artifact(test_artifact, "test_path")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and
        `trainer.test()`, so be careful not to execute the random split twice!
        The `stage` can be used to differentiate whether it's called before
        trainer.fit()`="fit" or `trainer.test()`="test".
        """
        self._set_data_path()

        if self.hparams.image_dim == 2:
            if self.hparams.unpaired_data:
                from cmrai.datamodules.components.brats2021 import (
                    BRATS2021Unpaired as BRATS2021,
                )
            else:
                from cmrai.datamodules.components.brats2021 import BRATS2021
        elif self.hparams.image_dim == 3:
            from cmrai.datamodules.components.brats2021_3d import BRATS2021
        else:
            raise ValueError("image_dim must be 2 or 3")

        # load datasets only if they're not loaded already
        if self.train_path and not self.data_train:
            self.data_train = BRATS2021(
                self.processed_path,
                self.train_path,
                train=True,
                transform=self.transforms_train,
                target_transform=self.transforms_target,
                final_transform=self.transforms_final,
            )
            log.info(f"\tTraining set size: {len(self.data_train)}")

        if self.val_path and not self.data_val:
            self.data_val = BRATS2021(
                self.processed_path,
                self.val_path,
                train=False,
                transform=self.transforms_val,
                final_transform=self.transforms_final,
            )
            log.info(f"\tValidation set size: {len(self.data_val)}")

        if self.test_path and not self.data_test:
            self.data_test = BRATS2021(
                self.processed_path,
                self.test_path,
                train=False,
                transform=self.transforms_val,
                final_transform=self.transforms_final,
            )
            log.info(f"\tTest set size: {len(self.data_test)}")

    def train_dataloader(self):
        _cfg_dict = {
            "dataset": self.data_train,
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.persistent_workers,
            "pin_memory": self.hparams.pin_memory,
            "shuffle": self.hparams.shuffle_training,
        }
        if self.hparams.prefetch_factor:
            _cfg_dict["prefetch_factor"] = self.hparams.prefetch_factor
        return LoaderWrapper(
            DataLoader(**_cfg_dict), n_step=self.hparams.n_step_per_epoch
        )

    def val_dataloader(self):
        _cfg_dict = {
            "dataset": self.data_val,
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.persistent_workers,
            "pin_memory": self.hparams.pin_memory,
            "shuffle": False,
        }
        if self.hparams.prefetch_factor:
            _cfg_dict["prefetch_factor"] = self.hparams.prefetch_factor
        return DataLoader(**_cfg_dict)

    def test_dataloader(self):
        _cfg_dict = {
            "dataset": self.data_test,
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.persistent_workers,
            "pin_memory": self.hparams.pin_memory,
            "shuffle": False,
        }
        if self.hparams.prefetch_factor:
            _cfg_dict["prefetch_factor"] = self.hparams.prefetch_factor
        return DataLoader(**_cfg_dict)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
