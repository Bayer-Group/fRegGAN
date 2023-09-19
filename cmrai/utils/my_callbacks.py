import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Union

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sn
import seaborn_image as isns
import torch

# from aim import Image
from dill import load as dill_load
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

import wandb


def get_wandb_logger(trainer: Trainer) -> Union[WandbLogger, None]:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    return None


def get_aim_logger(trainer: Trainer) -> Union["AimLogger", None]:
    """Safely get aim logger from Trainer."""
    return None

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, "AimLogger"):
        return trainer.logger

    return None


class UploadToGCS(Callback):
    """Upload log folder to gs bucket."""

    def __init__(
        self,
        folder_to_upload: str,
    ):
        self.folder_to_upload = Path(folder_to_upload)
        self.bucket_folder = (
            f"gs://{os.environ['GC_BUCKET']}/{os.environ['GC_USER']}-data/"
        )
        self.bucket_folder = self.bucket_folder + str(
            self.folder_to_upload.relative_to(Path(os.environ["PROJECT_ROOT"]))
        )

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self._upload()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self._upload()

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        self._upload()

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        self._upload()

    def _upload(self):
        run_cmd = f"gsutil -m rsync -d -r {self.folder_to_upload} {self.bucket_folder}"
        subprocess.run(run_cmd, shell=True)


class LogMetricsToWandb(Callback):
    """Log metrics to wandb."""

    def __init__(self, test_artefact_name: str):
        # create a wandb Artifact for each meaningful step
        self.test_artefact_name = test_artefact_name
        self.test_metric_artifact = {}
        self.wandb_table: wandb.Table = None
        self.idx_counter = 0

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        metric_dict: dict = outputs["batch_results"]  # type: ignore
        if not self.wandb_table:
            col_names = ["sample_id", "image_path"]
            for metric_name in metric_dict.keys():
                col_names += [metric_name]

            self.wandb_table = wandb.Table(columns=col_names)
            self.test_metric_artifact = wandb.Artifact(
                f"{self.test_artefact_name}", type=f"metrics"
            )

        for idx in range(batch[0].shape[0]):
            # convert idx to globle indexto get file name
            image_path = trainer.datamodule.data_test.data[self.idx_counter]  # type: ignore
            sample_id = image_path.split("/")[0]
            new_row = [sample_id, image_path]

            for k in metric_dict.keys():
                metric_result = metric_dict[k]
                if isinstance(metric_result, torch.Tensor):
                    metric_result = metric_result.tolist()
                new_row += [float(metric_result[idx])]
            self.wandb_table.add_data(*new_row)
            self.idx_counter += 1

    @rank_zero_only
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test ends."""
        wandb_logger = wandb.run
        if not wandb_logger:
            raise Exception("Wandb logger not found.")

        self.test_metric_artifact.add(self.wandb_table, "metrics_results")  # type: ignore
        wandb_logger.log_artifact(self.test_metric_artifact)

        # TODO: implement better result logging! Figures and tables
        # aggregate metrics over all samples and log them
        # metric_dict = {}
        # metric_names = set()
        # for _, row in self.wandb_table.iterrows():
        #     for idx, k in enumerate(self.wandb_table.columns):
        #         metric_dict[k].append(row[idx])

        # df = pd.DataFrame(data=metric_dict)
        # for metric_name in metric_names:
        #     df_metric = df[df["metric"] == metric_name]
        #     wandb.run.summary[metric_name + "_mean"] = df_metric["value"].mean()
        #     wandb.run.summary[metric_name + "_std"] = df_metric["value"].std()


class LogNiiImages(Callback):
    """_summary_

    Args:
        Callback (_type_): _description_
    """

    def __init__(
        self,
        output_dir: str,
        images_to_save: int,
        image_name: str,
        save_deformation_field: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_to_save = images_to_save
        self.image_name = image_name
        self.save_deformation_field = save_deformation_field
        self.image_counter = 0
        self.last_sample_name = None
        self.image_slices = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def _save_image(self, trainer, slice_id_current):
        last_sample_path = Path(trainer.datamodule.data_test.data[slice_id_current - 1])
        # find regex "dim-([0-9]+)" in self.last_sample_name
        axis_dim = int(re.search("dim-([0-9]+)", str(self.last_sample_name)).group(1))

        if self.save_deformation_field:
            img_3d = np.stack(self.image_slices, axis=axis_dim + 1)
            np.save(
                str(self.output_dir / f"{self.last_sample_name}.npy"),
                np.transpose(img_3d, (1, 2, 3, 0)),
            )
            self.image_counter += 1
            self.image_slices = []
        else:
            nifti_header = nib.Nifti1Header()
            nifti_affine = None
            with h5py.File(
                str(trainer.datamodule.data_test.hdf5_file), "r"
            ) as hdf5_file:
                nifti_meta_data = hdf5_file[str(last_sample_path.parent)].attrs
                for key, value in nifti_meta_data.items():
                    if "nib.header_" in key:
                        # remove nib.header_ prefix
                        # and set value to nifti header
                        setattr(nifti_header, "_".join(key.split("_")[1:]), value)
                nifti_affine = nifti_meta_data["nib.affine"]

            img_3d = np.stack(self.image_slices, axis=axis_dim)

            # convert to nifti
            nib_img3d = nib.Nifti1Image(img_3d, nifti_affine, nifti_header)
            nib.save(
                nib_img3d, str(self.output_dir / f"{self.last_sample_name}.nii.gz")
            )

            self.image_counter += 1
            self.image_slices = []

    @rank_zero_only
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.images_to_save < 0 or self.image_counter < self.images_to_save:
            img = getattr(pl_module, f"last_{self.image_name}")
            is_3d = len(img) == 5
            batch_size = img.shape[0]
            sample_id = batch_idx * batch_size
            if is_3d:
                # TODO: update support for 3D -> hdf5
                raise NotImplementedError
                for idx_sample in range(batch_size):
                    sample_id_current = sample_id + idx_sample
                    data_path = trainer.datamodule.data_test.data[slice_id_current]
                    if "PigsMRI" in str(data_path):
                        data_name = Path(data_path).name
                        sample_name = (
                            "_".join(data_name.split("_")[:2])
                            + f"-{data_path.split('/')[-2]}"
                        )
                    else:
                        data_name = Path(data_path).name
                        sample_name = "_".join(data_name.split("_")[:2])
                    nib.save(
                        pl_module.last_fake_B[idx_sample],
                        str(self.output_dir / f"fake_{data_name}"),
                    )
                    if self.save_deformation_field:
                        nib.save(
                            pl_module.last_deformation_field_B[idx_sample],
                            str(self.output_dir / f"deformation_{data_name}"),
                        )
                    self.image_counter += 1
            else:
                for idx_slice in range(batch_size):
                    slice_id_current = sample_id + idx_slice
                    data_path = trainer.datamodule.data_test.data[slice_id_current]

                    sample_name = ("_").join(
                        Path(data_path).name.split("_")[:-1]
                    )  # remove slice number from name

                    # NOTE: this is a hack to get the correct block id for the PigsMRI dataset
                    if "KM20093" in str(data_path):
                        # find regex "Block([0-9]+)" in data_path
                        block_id = re.search("Block([0-9]+)", str(data_path)).group(0)
                        sample_name += f"_{block_id}"

                    # image is finished if sample name changes
                    if self.last_sample_name and self.last_sample_name != sample_name:
                        self._save_image(trainer, slice_id_current)

                    self.image_slices.append(np.squeeze(img[idx_slice]))
                    self.last_sample_name = sample_name

    def on_test_end(self, trainer, pl_module):
        # save last image
        self._save_image(trainer, len(trainer.datamodule.data_test.data))


class LogExampleImages(Callback):
    """_summary_

    Args:
        Callback (_type_): _description_
    """

    def __init__(
        self,
        name: str,
        image_list: list,
        output_dir: str,
        log_train_img_interval: int,
        log_val_img_interval: int,
        slice_axis: int = 3,
        img_dpi: int = 45,
        is_kspace_domain: bool = False,
    ):
        self.name = name
        self.image_list = image_list
        self.titles = [x for x in image_list] + [
            f"Diff({image_list[-2]} - {image_list[-1]})"
        ]
        self.output_dir = Path(output_dir) / "example_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(self.output_dir)
        self.log_train_img_interval = log_train_img_interval
        self.log_val_img_interval = log_val_img_interval
        self.slice_axis = slice_axis
        self.img_dpi = img_dpi
        self.ready = True
        self.is_kspace_domain = is_kspace_domain

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def inverse_data_normalize(
        self,
        img: npt.ArrayLike,
        data_mean: float,
        data_std: float,
        max_pixel_value: float,
    ) -> npt.ArrayLike:
        return (img * (data_std * max_pixel_value)) + data_mean

    @rank_zero_only
    def create_and_log_img(self, trainer, pl_module, batch_idx, phase):
        img_col = []
        for img_name in self.image_list:
            img_col.append(getattr(pl_module, f"last_{img_name}"))

        if phase == "test":
            img_col = [x[0][0] for x in img_col]  # remove batch and channel dim

        # check image domain
        if self.is_kspace_domain:
            if img_col[1].shape[0] == 2:
                img_col = [x[0] + 1j * x[1] for x in img_col]
            img_col = [np.abs(x) for x in img_col]
            img_col.append(img_col[-2] - img_col[-1])
            img_grid_kwargs = {
                "cmap": ["viridis", "viridis", "viridis", "vlag"],
                "cbar_log": [True, True, True, False],
                "cbar": [True, True, True, True],
                "showticks": True,
                "col_wrap": 4,
                "height": 5,
            }

        else:
            if phase != "test":
                # invert norm
                img_col = [
                    self.inverse_data_normalize(
                        img,
                        trainer.datamodule.hparams.data_mean,
                        trainer.datamodule.hparams.data_std,
                        trainer.datamodule.hparams.max_pixel_value,
                    )
                    for img in img_col
                ]
            img_col.append(img_col[-2] - img_col[-1])
            img_grid_kwargs = {
                "cmap": ["gray", "gray", "gray", "vlag"],
                "cbar": [True, True, True, True],
                "vmin": [0, 0, 0, -1],
                "vmax": [1, 1, 1, 1],
                "showticks": True,
                "col_wrap": 4,
                "height": 5,
            }

        # check dimensions image and slice if 3d
        if len(img_col[0].shape) == 3:
            mid_slice_idx = img_col[0].shape[-1] // 2
            img_col = [img[..., mid_slice_idx] for img in img_col]

        img_grid = isns.ImageGrid(img_col, **img_grid_kwargs)
        for ax, title in zip(img_grid.axes.ravel(), self.titles):
            ax.set_title(title)
        output_file_name = f"{self.output_dir}/{phase}_{self.name.split('/')[-1]}_{batch_idx}_{trainer.global_step}.png"
        img_grid.fig.savefig(output_file_name, dpi=self.img_dpi)
        plt.close(img_grid.fig)

        file_path = Path(output_file_name).absolute()
        if file_path is not None and file_path.is_file():
            wandb_logger = get_wandb_logger(trainer)
            if wandb_logger:
                wandb_logger.log_image(
                    key=f"{phase}/{self.name}/{batch_idx}",
                    images=[str(file_path)],
                )
            aim_logger = get_aim_logger(trainer)
            if aim_logger:
                aim_image = Image(
                    str(file_path), format="png", optimize=True, quality=50
                )
                log_name = f"{phase}/{self.name}/{batch_idx}"
                if phase == "train":
                    log_name = f"{phase}/{self.name}"
                aim_logger.log_image(
                    aim_image,
                    name=log_name,
                    step=trainer.fit_loop.epoch_loop._batches_that_stepped,
                    phase=phase,
                )
            # file_path.unlink(missing_ok=True)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.log_train_img_interval and batch_idx % self.log_train_img_interval == 0:
            self.create_and_log_img(trainer, pl_module, batch_idx, "train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.log_val_img_interval and batch_idx % self.log_val_img_interval == 0:
            self.create_and_log_img(trainer, pl_module, batch_idx, "val")

    @rank_zero_only
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.log_val_img_interval and batch_idx % self.log_val_img_interval == 0:
            self.create_and_log_img(trainer, pl_module, batch_idx, "test")
