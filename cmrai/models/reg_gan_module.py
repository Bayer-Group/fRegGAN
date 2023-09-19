import itertools
import os
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn_image as isns
import torch
import torchmetrics.functional as tmf
from kornia.filters import GaussianBlur2d
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchmetrics import (
    MaxMetric,
    MeanAbsoluteError,
    MeanMetric,
    MinMetric,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from cmrai.models import my_model_utils
from cmrai.utils import my_metric


class RegGanLitSystem(LightningModule):
    """RegGanLitSystem.

    A LightningSystem organizes our PyTorch code this order:
        - System definition (init).
        - Inference call (forward)
        - Training hooks:
            - (training_step)
            - (training_step_end)
            - (training_epoch_end)
        - Validation hooks
            - (validation_step)
            - (validation_step_end)
            - (validation_epoch_end)
        - Test hooks:
            - (test_step)
            - (test_step_end)
            - (test_epoch_end)
        - Prediction hooks:
            - (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
        - Any other hooks (any_extra_hook)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        # Models
        net_generator_A2B: torch.nn.Module,
        net_discriminator_B: torch.nn.Module,
        # Optimizer stuff
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        p2p_lamda: float,
        # Extra stuff
        net_deformation_field_B: torch.nn.Module = None,
        spatial_deformation_B: torch.nn.Module = None,
        identity_lamda: float = 0.0,
        correction_lamda: float = 0,
        smooth_lamda: float = 0,
        freq_lamda: float = 0,
        freq_weighting: float = 0,
        freq_radius: float = 0,
        lr_scheduler: bool = False,
        save_graph_onnx: bool = False,
        metric_on_regist: bool = False,
        num_interativ_steps=0,
        **kwargs,
    ):
        super().__init__()
        self.net_generator_A2B = net_generator_A2B
        self.net_discriminator_B = net_discriminator_B

        self.net_deformation_field_B = net_deformation_field_B
        self.spatial_deformation_B = spatial_deformation_B

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        ignore_list = []
        for k in self._modules.keys():
            ignore_list.append(k)
        self.save_hyperparameters(logger=False, ignore=ignore_list)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_MAE = MeanAbsoluteError()
        for stage in ["val", "test"]:
            setattr(self, f"{stage}_MAE", MeanAbsoluteError())
            setattr(self, f"{stage}_SSIM", MeanMetric())
            setattr(self, f"{stage}_PSNR", PeakSignalNoiseRatio())

            # for logging best so far
            setattr(self, f"{stage}_MAE_best", MinMetric())
            setattr(self, f"{stage}_SSIM_best", MaxMetric())
            setattr(self, f"{stage}_PSNR_best", MaxMetric())

        self.fakePoolA = my_model_utils.ImagePool()
        self.fakePoolB = my_model_utils.ImagePool()

        self.example_input_array = {}
        # TODO: remove magic numbers!!!
        example_image = torch.zeros(
            (2, self.net_generator_A2B.input_channels)
            + self.net_generator_A2B.input_size
        ).float()

        for m in ignore_list:
            self.example_input_array[m] = example_image

    def get_mse_loss(self, predictions, label):
        """According to the CycleGan paper, label for real is one and fake is zero."""
        if label.lower() == "real":
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)

        return F.mse_loss(predictions, target)

    def forward(self, *args, **kwargs):
        """Forward pass.

        You should treat this as the final inference code
        (final outputs that you are looking for!), but you can definitely use it in the training_step
        to make some code reusable.
        Parameters:
            real_A -- real image of A
        """
        if self.example_input_array:
            for key, value in kwargs.items():
                if "net_deformation_field" in key:
                    deform = getattr(self, key)(value, value)
                elif "spatial_deformation" in key:
                    r = getattr(self, key)(value, deform)
                else:
                    r = getattr(self, key)(value)
            return r
        # check if real_A is provides as positional argument or as keyword argument
        if len(args) > 0:
            real_A = args[0]
        else:
            real_A = kwargs["real_A"]
        fake_B = self.net_generator_A2B(real_A)
        return fake_B

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch
        discriminator_requires_grad = optimizer_idx == 1
        my_model_utils.set_requires_grad(
            [self.net_discriminator_B],
            discriminator_requires_grad,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        # return {"loss": loss, "preds": preds, "targets": targets}
        if optimizer_idx == 0:
            return self.generator_training_step(real_A, real_B, batch_idx)
        else:
            return self.discriminator_training_step(real_A, real_B)

    def generator_training_step(self, real_A, real_B, batch_idx):
        fake_B = self(real_A=real_A)  # G_A2B(A)
        identity_B = self.net_generator_A2B(real_B)  # G_A2B(B)

        if self.net_deformation_field_B:
            deformation_field_B = self.net_deformation_field_B(fake_B, real_B)
            regist_fake_B = self.spatial_deformation_B(fake_B, deformation_field_B)

        pred_fake_B = self.net_discriminator_B(fake_B)  # D_B(G_A2B(A))

        # generator net_G_A2B must fool discrim net_D_B so label is real = 1
        # GAN loss D_B(G_A2B(A))
        generator_A2B_loss = self.get_mse_loss(pred_fake_B, "real")

        # G_A2B should be identity if real_B is fed: ||G_A2B(B) - B||
        identity_loss_B = 0
        if self.hparams.identity_lamda > 0:
            identity_loss_B = F.l1_loss(identity_B, real_B)

        correction_loss = 0
        smoothing_loss = 0
        if self.net_deformation_field_B:
            correction_loss, smoothing_loss = self.get_correction_loss(
                real_B,
                regist_fake_B,
                deformation_field_B,
            )
            loss_L1 = 0
        else:
            loss_L1 = F.l1_loss(fake_B, real_B) * self.hparams.p2p_lamda

        freq_loss = 0
        if self.hparams.freq_lamda > 0:
            freq_loss = my_model_utils.fft2d_loss(
                real_B,
                fake_B if not self.net_deformation_field_B else regist_fake_B,
                self.hparams.freq_radius,
                self.hparams.freq_weighting,
                self.hparams.use_euclid,
            )

        # Total loss
        self.generator_loss = (
            generator_A2B_loss
            + loss_L1
            + self.hparams.identity_lamda * identity_loss_B
            + self.hparams.correction_lamda * correction_loss
            + self.hparams.smooth_lamda * smoothing_loss
            + self.hparams.freq_lamda * freq_loss
        )

        # store detached generated images
        self.fake_B = fake_B.detach()

        # log losses
        dict_ = {
            "train/gen_total_loss": self.generator_loss,
            "train/gen_A2B_loss": generator_A2B_loss,
            "train/p2p_loss": loss_L1,
        }
        if self.hparams.identity_lamda > 0:
            dict_["train/identity_loss"] = identity_loss_B
        if self.net_deformation_field_B:
            dict_["train/correction_loss"] = (
                self.hparams.correction_lamda * correction_loss
                + self.hparams.smooth_lamda * smoothing_loss
            )
        if self.hparams.freq_lamda > 0:
            dict_["train/freq_loss"] = self.hparams.freq_lamda * freq_loss

        self.log_dict(
            dict_,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/gen_loss",
            self.generator_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )

        if self.net_deformation_field_B:
            # log train metrics
            mae = self.train_MAE(regist_fake_B, real_B)
            self.log(
                "train/mae",
                mae,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        idx = 0
        self.last_real_A = np.squeeze(real_A[idx].detach().cpu().numpy())
        self.last_real_B = np.squeeze(real_B[idx].detach().cpu().numpy())
        self.last_fake_B = np.squeeze(fake_B[idx].detach().cpu().numpy())

        if self.net_deformation_field_B:
            self.last_reg_fake_B = np.squeeze(regist_fake_B[idx].detach().cpu().numpy())

        return self.generator_loss

    def get_correction_loss(
        self,
        real,
        regist_fake,
        deformation_field,
    ):
        correction_loss = F.l1_loss(regist_fake, real)
        smoothing_loss = my_model_utils.smoothing_loss(deformation_field)
        return correction_loss, smoothing_loss

    def discriminator_training_step(self, real_A, real_B):
        """Update Discriminator."""
        fake_B = self.fakePoolB.query(self.fake_B)

        # net_D_B checks for domain B photos
        pred_real_B = self.net_discriminator_B(real_B)
        pred_fake_B = self.net_discriminator_B(fake_B)

        mse_real_B = self.get_mse_loss(pred_real_B, "real")
        mse_fake_B = self.get_mse_loss(pred_fake_B, "fake")

        self.discriminator_loss = 0.5 * (mse_fake_B + mse_real_B)
        self.log(
            "train/dis_loss",
            self.discriminator_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return self.discriminator_loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        real_A, real_B = batch

        fake_B = self(real_A=real_A)  # G_A2B(A)
        identity_B = self.net_generator_A2B(real_B)  # G_A2B(B)

        if self.net_deformation_field_B:
            deformation_field_B = self.net_deformation_field_B(fake_B, real_B)
            regist_fake_B = self.spatial_deformation_B(fake_B, deformation_field_B)

        pred_real_B = self.net_discriminator_B(real_B)  # D_B(B)
        pred_fake_B = self.net_discriminator_B(fake_B)  # D_B(G_A2B(A))

        # log loss
        generator_A2B_loss = self.get_mse_loss(pred_fake_B, "real")
        loss_L1 = F.l1_loss(fake_B, real_B) * self.hparams.p2p_lamda

        # G_A2B should be identity if real_B is fed: ||G_A2B(B) - B||
        identity_loss_B = 0
        if self.hparams.identity_lamda > 0:
            identity_loss_B = F.l1_loss(identity_B, real_B)

        correction_loss = 0
        smoothing_loss = 0
        if self.net_deformation_field_B:
            correction_loss, smoothing_loss = self.get_correction_loss(
                real_B,
                regist_fake_B,
                deformation_field_B,
            )

        freq_loss = 0
        if self.hparams.freq_lamda > 0:
            freq_loss = my_model_utils.fft2d_loss(
                real_B,
                fake_B if not self.net_deformation_field_B else regist_fake_B,
                self.hparams.freq_radius,
                self.hparams.freq_weighting,
                self.hparams.use_euclid,
            )

        # Total loss
        generator_loss = (
            generator_A2B_loss
            + loss_L1
            + self.hparams.identity_lamda * identity_loss_B
            + self.hparams.correction_lamda * correction_loss
            + self.hparams.smooth_lamda * smoothing_loss
            + self.hparams.freq_lamda * freq_loss
        )

        mse_real_B = self.get_mse_loss(pred_real_B, "real")
        mse_fake_B = self.get_mse_loss(pred_fake_B, "fake")
        discriminator_loss = 0.5 * (mse_real_B + mse_fake_B)

        # Log losses
        dict_ = {
            f"val/gen_total_loss": generator_loss,
            f"val/gen_A2B_loss": generator_A2B_loss,
            f"val/dis_total_loss": discriminator_loss,
        }
        if self.hparams.identity_lamda > 0:
            dict_[f"val/identity_loss"] = identity_loss_B
        if self.net_deformation_field_B:
            dict_[f"val/correction_loss"] = (
                self.hparams.correction_lamda * correction_loss
                + self.hparams.smooth_lamda * smoothing_loss
            )
        if self.hparams.freq_lamda > 0:
            dict_[f"val/freq_loss"] = self.hparams.freq_lamda * freq_loss
        self.log_dict(
            dict_,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        dict_ = {
            f"val/gen_loss": generator_loss,
            f"val/dis_loss": discriminator_loss,
        }
        self.log_dict(
            dict_,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )

        # log metrics
        if self.net_deformation_field_B and self.hparams.metric_on_regist:
            if regist_fake_B.type() != real_B.type():
                # need to convert to same type as real_B, if traind with mixed precision
                regist_fake_B = regist_fake_B.type_as(real_B)

            getattr(self, f"val_PSNR")(regist_fake_B, real_B)
            ssim_batch = tmf.structural_similarity_index_measure(
                regist_fake_B.detach().cpu(), real_B.detach().cpu(), data_range=2.0
            )
            getattr(self, f"val_SSIM")(ssim_batch)
            getattr(self, f"val_MAE")(regist_fake_B, real_B)
        else:
            if fake_B.type() != real_B.type():
                # need to convert to same type as real_B, if traind with mixed precision
                fake_B = fake_B.type_as(real_B)

            getattr(self, f"val_PSNR")(fake_B, real_B)
            ssim_batch = tmf.structural_similarity_index_measure(
                fake_B.detach().cpu(), real_B.detach().cpu(), data_range=2.0
            )
            getattr(self, f"val_SSIM")(ssim_batch)
            getattr(self, f"val_MAE")(fake_B, real_B)
        self.log(
            f"val/psnr",
            getattr(self, f"val_PSNR"),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/ssim",
            getattr(self, f"val_SSIM"),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/mae",
            getattr(self, f"val_MAE"),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        idx = 0
        self.last_real_A = np.squeeze(real_A[idx].detach().cpu().numpy())
        self.last_real_B = np.squeeze(real_B[idx].detach().cpu().numpy())
        self.last_fake_B = np.squeeze(fake_B[idx].detach().cpu().numpy())

        if self.net_deformation_field_B:
            self.last_regist_fake_B = np.squeeze(
                regist_fake_B[idx].detach().cpu().numpy()
            )

    def validation_epoch_end(self, outputs: List[Any]):
        mae = getattr(self, f"val_MAE").compute()
        psnr = getattr(self, f"val_PSNR").compute()
        ssim = getattr(self, f"val_SSIM").compute()
        getattr(self, f"val_MAE_best")(mae)
        getattr(self, f"val_PSNR_best")(psnr)
        getattr(self, f"val_SSIM_best")(ssim)
        _dict = {
            f"val/MAE_best": getattr(self, f"val_MAE_best").compute(),
            f"val/PSNR_best": getattr(self, f"val_PSNR_best").compute(),
            f"val/SSIM_best": getattr(self, f"val_SSIM_best").compute(),
        }
        # log `*_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # when logging as a value, set `sync_dist=True` for proper reduction over processes in DDP mode
        self.log_dict(
            _dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def calculate_metrics(self, fake_B, real_B):
        metric_list_tmr = {
            "universal_image_quality_index": {"reduction": "none", "data_range": 1},
            "structural_similarity_index_measure": {
                "reduction": "none",
                "data_range": 1,
            },
            # "spectral_distortion_index",
            # "spectral_angle_mapper",
            "peak_signal_noise_ratio": {
                "reduction": "none",
                "dim": (1, 2, 3),
                "data_range": 1,
            },
            "multiscale_structural_similarity_index_measure": {
                "reduction": "none",
                "data_range": 1,
            },
            "error_relative_global_dimensionless_synthesis": {"reduction": "none"},
        }

        metric_dict = {}
        for key, value in metric_list_tmr.items():
            metric_values = getattr(tmf, key)(fake_B, real_B, **value)
            if len(metric_values.shape) == 4:
                # mean over image
                metric_values = metric_values.nanmean(dim=[1, 2, 3])
            metric_dict[key] = metric_values
            # # log metric value per slice; step is used as slice index
            # for idx in range(len(metric_values)):
            #     self.logger.experiment.log_metrics(
            #         {key: metric_values[idx]}, step=batch_idx * real_A.shape[0] + idx
            #     )
        real_B = real_B.detach().cpu().numpy().squeeze()
        fake_B = fake_B.detach().cpu().numpy().squeeze()

        metric_dict["peak_signal_noise_ratio_masked"] = my_metric.psnr_masked(
            fake_B, real_B
        )
        metric_dict["mean_absolut_error"] = my_metric.mae(fake_B, real_B)
        metric_dict["mean_absolut_error_masked"] = my_metric.mae_masked(fake_B, real_B)

        return metric_dict

    def test_step(self, batch, batch_idx):
        real_A, real_B = batch
        fake_B = self(real_A=real_A)  # G_A2B(A)
        regist_fake_B = None
        if self.hparams.num_interativ_steps > 0:
            # create the operator
            gauss = GaussianBlur2d((3, 3), (1.0, 1.0))

            for idx in range(self.hparams.num_interativ_steps):
                x_blur: torch.tensor = gauss(fake_B)
                x_sharp: torch.tensor = fake_B + 1.0 * (fake_B - x_blur)
                fake_B = self(real_A=x_sharp)  # G_A2B(A)

        if self.net_deformation_field_B:
            deformation_field_B = self.net_deformation_field_B(fake_B, real_B)
            regist_fake_B = self.spatial_deformation_B(fake_B, deformation_field_B)
            # print(deformation_field_B.min(), deformation_field_B.max())

        # invert the image normalization
        data_mean = self.trainer.datamodule.hparams.data_mean
        data_std = self.trainer.datamodule.hparams.data_std

        real_A = real_A * data_std + data_mean  # [-1, 1] -> [0, 1]
        real_B = real_B * data_std + data_mean  # [-1, 1] -> [0, 1]
        fake_B = fake_B * data_std + data_mean  # [-1, 1] -> [0, 1]
        if self.net_deformation_field_B:
            regist_fake_B = regist_fake_B * data_std + data_mean  # [-1, 1] -> [0, 1]

        self.last_real_A = real_A.detach().cpu().numpy()
        self.last_real_B = real_B.detach().cpu().numpy()
        self.last_fake_B = fake_B.detach().cpu().numpy()
        if self.net_deformation_field_B:
            self.last_regist_fake_B = regist_fake_B.detach().cpu().numpy()
            self.last_deformation_field_B = deformation_field_B.detach().cpu().numpy()

        if self.hparams.metric_on_regist and self.net_deformation_field_B:
            metric_dict = self.calculate_metrics(regist_fake_B, real_B)
        else:
            metric_dict = self.calculate_metrics(fake_B, real_B)

        return {"batch_results": metric_dict}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        opt_d = self.hparams.optimizer_d(params=self.net_discriminator_B.parameters())

        arg_list = [
            self.net_generator_A2B.parameters(),
        ]
        if self.net_deformation_field_B:
            arg_list.append(self.net_deformation_field_B.parameters())
        g_params = itertools.chain(*arg_list)
        opt_g = self.hparams.optimizer_g(params=g_params)

        return [opt_g, opt_d], []

    def on_train_start(self):
        self.example_input_array = None

    def on_test_start(self):
        self.example_input_array = None


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "cycle_gan.yaml")
    _ = hydra.utils.instantiate(cfg)
