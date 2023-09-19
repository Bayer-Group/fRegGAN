import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from cmrai.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.
    Makes multirun more resistant to failure.
    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(config: DictConfig):

        # apply extra utilities
        extras(config)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(config=config)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(config.paths.output_dir, "exec_time.log")
            content = (
                f"'{config.task_name}' execution time: {time.time() - start_time} (s)"
            )
            save_file(
                path, content
            )  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {config.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(config: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not config.get("extras"):
        log.warning("Extras config not found! <config.extras=null>")
        return

    # disable python warnings
    if config.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if config.extras.get("enforce_tags"):
        log.info("Enforcing tags! <config.extras.enforce_tags=True>")
        rich_utils.enforce_tags(config, save_to_file=True)

    # pretty print config tree using Rich library
    if config.extras.get("print_config"):
        log.info("Printing config tree with Rich! <config.extras.print_config=True>")
        rich_utils.print_config_tree(config, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_config: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_config:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_config, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_config.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_config: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_config:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_config.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> Optional[dict]:
    """Controls which config parts are saved by lightning loggers. Additionally saves:

    - Number of model parameters
    """

    hparams = {}

    config = object_dict["config"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]
    datamodule = object_dict["datamodule"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["datamodule/testdata_size"] = (
        len(datamodule.data_test) if datamodule.data_test else None
    )
    hparams["datamodule/valdata_size"] = (
        len(datamodule.data_val) if datamodule.data_val else None
    )
    hparams["datamodule/traindata_size"] = (
        len(datamodule.data_train) if datamodule.data_train else None
    )

    hparams["trainer"] = config["trainer"]

    hparams["callbacks"] = config.get("callbacks")
    hparams["extras"] = config.get("extras")

    hparams["task_name"] = config.get("task_name")
    hparams["tags"] = config.get("tags")
    hparams["ckpt_path"] = config.get("ckpt_path")
    hparams["seed"] = config.get("seed")

    hydra_cfg = HydraConfig.get()
    hydra_dict = OmegaConf.to_container(hydra_cfg.runtime.choices)
    hparams["config_names"] = hydra_dict

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
    return hparams


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during
    multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
