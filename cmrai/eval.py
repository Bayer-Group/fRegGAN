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
# --------------------------------------------------------------------------------------s

import os

import hydra
from omegaconf import DictConfig

import cmrai.utils.my_resolver as my_resolver


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    from cmrai.tasks.eval_task import evaluate

    if cfg.logger.get("wandb", None):
        os.environ["WANDB_PROJECT"] = cfg.logger.wandb.project

    if cfg.get("ckpt_artefact", None) is None:
        # if no ckpt_artefact is specified, use wandb to find run with experiment config name
        from pathlib import Path

        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        import wandb

        api = wandb.Api()
        hydra_cfg = HydraConfig.get()
        hydra_dict = OmegaConf.to_container(hydra_cfg.runtime.choices)
        exp_name = Path(hydra_dict["experiment"]).with_suffix("")
        runs = api.runs(
            os.environ["WANDB_ENTITY"] + "/" + os.environ["WANDB_PROJECT"],
            {
                "config.config_names/experiment": {
                    "$regex": f"({exp_name}|{exp_name}.yaml)"
                }
            },
        )
        if len(runs) == 0:
            raise ValueError(f"No runs found for config {exp_name}")
        elif len(runs) > 1:
            raise ValueError(f"Multiple runs found for config {exp_name}")
        else:
            for a in runs[0].logged_artifacts():
                if a.type == "model":
                    cfg.ckpt_artefact = a.name
                    break

    evaluate(cfg)


if __name__ == "__main__":
    main()
