import argparse
import logging
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyrootutils
from icecream import ic
from mpire import WorkerPool
from natsort import index_natsorted, order_by_index
from sklearn.model_selection import RepeatedKFold, train_test_split

import wandb

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

log = logging.getLogger(__name__)


def do_processing(shared_data, file_path):
    if np.array(shared_data[0][file_path]).mean() != 0:
        return True
    return False


def do_processing_ratio(shared_data, file_path):
    data = np.array(shared_data[0][file_path])
    data_ration = 1 - (data == 0).sum() / (data.shape[0] * data.shape[1])
    if data_ration > 0.10:
        return True
    return False


def create_splitting_file(
    idx_list, hdf5_file, list_sample_ids, cv, r, phase, data_name
):

    x = []
    y = []
    for idx in idx_list:
        sample_id = list_sample_ids[idx]
        for file_name in list(hdf5_file[f"{sample_id}/t1"].keys()):
            x.append(f"{sample_id}/t1/{file_name}")
            y.append(f"{sample_id}/t2/{file_name}")

    index = index_natsorted(x)
    x = order_by_index(x, index)
    y = order_by_index(y, index)
    if args.data_dim == "2d":
        if args.do_cleaning or args.do_cleaning_ratio:
            if args.do_cleaning:
                fct = do_processing
            else:
                fct = do_processing_ratio

            # removing all empty slices!
            with WorkerPool(
                n_jobs=16, daemon=False, shared_objects=(hdf5_file,)
            ) as pool:
                results = pool.map(fct, x, iterable_len=len(x), progress_bar=True)
            x = list(np.array(x)[results])
            y = list(np.array(y)[results])

    print(f"Length of {phase}_cv{cv}_rep{r}: {len(x)}")
    with open(
        root / DATA_ROOT / f"{data_name}_data_{phase}_cv{cv}_rep{r}.txt",
        "w",
    ) as fp:
        for p_img, p_label in zip(x, y):
            if Path(p_img).name == Path(p_label).name:
                fp.write(f"{p_img} {p_label}\n")
            else:
                print("Skipping !!")
    return len(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="latest",
        help="Version of the dataset to download. Defaults to latest.",
    )
    parser.add_argument(
        "--data_dim",
        type=str,
        help="Select '2d' or '3d' for sliced or volume data.",
    )
    parser.add_argument(
        "--do_cleaning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, remove slices with no information.",
    )
    parser.add_argument(
        "--do_cleaning_ratio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, remove slices with no information.",
    )
    return parser.parse_args()


args = parse_args()
DATA_ROOT = "data/BRATS2021"

if __name__ == "__main__":
    run = wandb.init(project="cmrai-brats2021-test", job_type="data_splitting")
    data_at: wandb.Artifact = run.use_artifact(
        f"BRATS2021_{args.data_dim}ClipNorm:{args.dataset_version}",
        type="processed_data",
    )

    hdf5_file_path = root / DATA_ROOT / "processed" / f"{args.data_dim}_ClipNorm.h5"

    print("Downloading dataset version with id: ", args.dataset_version)
    os.remove(hdf5_file_path)
    data_at.download(root=str(hdf5_file_path.parent))

    datatype = f"BRATS2021-{args.data_dim}"
    hf = h5py.File(str(hdf5_file_path), "r")

    print("Getting list of unique sample ids...")
    list_sample_ids = np.array(list(hf.keys()))

    random_state = 12883823
    num_splits = 5
    n_repeats = 1

    rkf = RepeatedKFold(
        n_splits=num_splits, n_repeats=n_repeats, random_state=random_state
    )
    dataset_length = []
    for r, (train, test_val) in enumerate(rkf.split(list_sample_ids)):
        cv = r % num_splits
        r = r // num_splits
        ic(len(list_sample_ids[train]))
        ic(len(list_sample_ids[test_val]))
        train_len = create_splitting_file(
            train, hf, list_sample_ids, cv, r, phase="train", data_name=datatype
        )

        val, test = train_test_split(
            test_val, test_size=0.80, shuffle=False, random_state=random_state
        )
        ic(len(list_sample_ids[test]))
        ic(len(list_sample_ids[val]))
        val_len = create_splitting_file(
            val, hf, list_sample_ids, cv, r, phase="val", data_name=datatype
        )

        test_len = create_splitting_file(
            test, hf, list_sample_ids, cv, r, phase="test", data_name=datatype
        )
        dataset_length.append([train_len, val_len, test_len, (cv, r)])
        break

    for train_len, val_len, test_len, (cv, r) in dataset_length:
        for phase in ["train", "val", "test"]:
            if phase == "train":
                length = train_len
            elif phase == "val":
                length = val_len
            else:
                length = test_len

            file_name = f"{datatype}_data_{phase}_cv{cv}_rep{r}.txt"
            dataset_at = wandb.Artifact(
                f"{datatype}_{phase}_{length}",
                type=f"{phase}_data",
                metadata={"subfolder": str(DATA_ROOT)},
            )
            dataset_at.add_file(str(root / DATA_ROOT / file_name))
            run.log_artifact(dataset_at)

    wandb.finish()
