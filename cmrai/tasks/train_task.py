import os
from pathlib import Path
from typing import List, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

import wandb
from cmrai import utils
from cmrai.utils import my_callbacks
from wandb import wandb_run, wandb_sdk

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(config: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights
    obtained during training.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(config.get("logger"))

    if wandb.run:
        wandb.run.log_code(os.path.join(os.environ["PROJECT_ROOT"], "cmrai"))

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule,
        _convert_=None,
        _recursive_=False,
    )
    datamodule.setup(stage="initial")

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(config.get("callbacks"))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    object_dict = {
        "config": config,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        cfg_dict = utils.log_hyperparameters(object_dict)

    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path")
        )

        # Save best model reference to Wandb
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if wandb.run and ckpt_path != "":
            log.info(f"Best ckpt path: {ckpt_path}")
            model_at = wandb_sdk.wandb_artifacts.Artifact(
                name=cfg_dict["task_name"], type="model", metadata=cfg_dict
            )
            model_at.add_file(ckpt_path)
            wandb.run.log_artifact(model_at)

    train_metrics = trainer.callback_metrics

    # Test the model
    if config.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        else:
            log.info(f"Best ckpt path: {ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict
