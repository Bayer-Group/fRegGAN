import pyrootutils

# --------------------------------------------------------------------------------------
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
#
# searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable
# used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# - move this file to the project root dir
# - always run this file from the project root dir
# - modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
#
# --------------------------------------------------------------------------------------
import os

import hydra
from omegaconf import DictConfig

import cmrai.utils.my_resolver as my_resolver


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float:
    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from cmrai.tasks.train_task import train
    from cmrai.utils import get_metric_value

    if cfg.logger and cfg.logger.get("wandb", None):
        os.environ["WANDB_PROJECT"] = cfg.logger.wandb.project

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
