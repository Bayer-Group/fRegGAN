import argparse
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pyrootutils
from dill import dump
from google.cloud import storage
from icecream import ic
from mpire import WorkerPool

import wandb

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


DATA_FOLDER = "data/BRATS2021/raw"
WANDB_PROJECT = "cmrai-brats2021-test"

if __name__ == "__main__":
    # TODO: check project if artefact already exists
    # https://docs.wandb.ai/guides/artifacts/overview
    # https://docs.wandb.ai/guides/artifacts/using-artifacts
    # https://docs.wandb.ai/guides/artifacts/using-artifacts#using-artifacts-in-python

    run = wandb.init(project=WANDB_PROJECT, job_type="upload")
    dataset_at: wandb.Artifact = wandb.Artifact(
        "BRATS2021", type="raw_data", metadata={"data_type": ".nii.gz", "dim": 3}
    )

    dataset_at.add_dir(os.path.join(root, DATA_FOLDER))
    run.log_artifact(dataset_at)
    wandb.finish()
