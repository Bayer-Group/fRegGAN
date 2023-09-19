import os
from pathlib import Path
from typing import List, Tuple

import hydra
from numpy import str0
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

import wandb
from cmrai import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(config: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert config.ckpt_artefact

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(config.get("logger"))

    if wandb.run:
        wandb.run.log_code(os.path.join(os.environ["PROJECT_ROOT"], "cmrai"))

    config.callbacks.log_metrics_to_wandb.test_artefact_name = (
        str(config.callbacks.log_metrics_to_wandb.test_artefact_name)
        .replace("-", "_")
        .split(":")[0]
    )

    assert wandb.run

    if ".ckpt" in config.ckpt_artefact:
        ckpt_path = config.ckpt_artefact
    else:
        # Download checkpoint from wandb to local output dir
        log.info("Downloading checkpoint from wandb...")
        ckpt_artifact = wandb.run.use_artifact(config.ckpt_artefact, type="model")
        file_name, _ = next(iter(ckpt_artifact.manifest.entries.items()))
        Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
        log.info(
            f"Downloading {file_name} with version id: {ckpt_artifact.name.split(':')[1]}"
        )
        (Path(config.paths.output_dir) / file_name).unlink(missing_ok=True)
        ckpt_artifact.download(root=config.paths.output_dir)
        ckpt_path = str(Path(config.paths.output_dir) / file_name)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule,
        _convert_=None,
        _recursive_=False,
    )

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

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if wandb.run:
        log.info(f"Saving image predictions as artifact to wandb...")
        images_at = wandb.Artifact(
            name=cfg_dict["task_name"]
            + "-"
            + str(config.datamodule.test_artefact).replace("-", "_").split(":")[0],
            type="predictions",
            metadata=cfg_dict,
        )
        images_at.add_dir(config.callbacks.save_nii_fake_B.output_dir)
        wandb.run.log_artifact(images_at)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=config.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict
