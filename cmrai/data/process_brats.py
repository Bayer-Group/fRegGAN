import argparse
import os
import re
import shutil
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import numpy.typing as npt
import pyrootutils
import skimage
from dill import dump
from google.cloud import storage
from icecream import ic
from mpire import WorkerPool
from tqdm import tqdm

import wandb
from wandb import wandb_run, wandb_sdk

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def slice_and_save_img(shared_data, axis, idx, data_dir, sample_id):
    """TODO: docstring

    Args:
        shared_data (_type_): _description_
        axis (_type_): _description_
        idx (_type_): _description_
        data_dir (_type_): _description_
        sample_id (_type_): _description_
    """
    img = shared_data[0]
    if axis == 0:
        img_out = img[idx, :, :]
    elif axis == 1:
        img_out = img[:, idx, :]
    else:
        img_out = img[:, :, idx]

    np.save(f"{data_dir}/{sample_id}_dim-{axis}_slice-{idx:04}.npy", img_out)


def clip_and_norm(
    img3d: npt.NDArray[np.float32], low_percentile: float, high_percentile: float
) -> npt.NDArray[np.float32]:
    """Clip and normalize image.

    Args:
        img3d (npt.NDArray[np.float64]): Image to clip and normalize.
        low_percentile (float): Lower percentile.
        high_percentile (float): Higher percentile.

    Returns:
        npt.NDArray[np.float64]: Clipped and normalized image.
    """
    p_low = np.percentile(img3d, low_percentile)
    p_high = np.percentile(img3d, high_percentile)
    img3d = np.clip(img3d, p_low, p_high)
    # norm float data to [0, 1]
    img3d = (img3d - img3d.min()) / (img3d.max() - img3d.min())
    return img3d


def do_hdf5_processing(filepath: Path, hf: h5py.File, axis=2):
    """Process the data for HDF5 format."""
    nib_img3d: nib.Nifti1Image = nib.load(str(filepath))
    img3d = nib_img3d.get_fdata().astype(np.float32)
    img_shape = img3d.shape

    img3d = clip_and_norm(img3d, 0.5, 99.9)

    # convert from 3D to 2D slices (each direction)
    filename = str(filepath.name)
    sample_id = filename.split("_")[1]
    if "_t1." in filename:
        sub_folder = sample_id + "/t1"
    else:
        sub_folder = sample_id + "/t2"

    hdf5_grp = hf.require_group(sub_folder)
    if args.data_dim == "2d":
        for idx in range(img_shape[axis]):
            if axis == 0:
                img_out = img3d[idx, :, :]
            elif axis == 1:
                img_out = img3d[:, idx, :]
            else:
                img_out = img3d[:, :, idx]
            data_name = f"{sample_id}_dim-{axis}_slice-{idx:04}"
            hdf5_grp.create_dataset(data_name, data=img_out, compression="gzip")
    else:
        data_name = f"{sample_id}"
        hdf5_grp.create_dataset(data_name, data=img3d, compression="gzip")

    for k, v in dict(nib_img3d.header).items():
        hdf5_grp.attrs[f"nib.header_{k}"] = v
    hdf5_grp.attrs["nib.affine"] = nib_img3d.affine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="latest",
        help="Version of the dataset to download. Defaults to latest.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, force to rerun data processing.",
    )
    parser.add_argument(
        "--data_dim",
        type=str,
        default="2d",
        help="Select 2d or 3d for sliced or volume data.",
    )
    return parser.parse_args()


args = parse_args()

DATA_ROOT = "data/BRATS2021"
OUTPUT_FOLDER = root / DATA_ROOT / "processed" / f"{args.data_dim}_ClipNorm.h5"

if __name__ == "__main__":
    run: wandb_run.Run = wandb.init(project="cmrai-brats2021-test", job_type="data_processing", force=True)  # type: ignore
    data_at: wandb_sdk.wandb_artifacts.Artifact = run.use_artifact(
        f"BRATS2021:{args.dataset_version}", type="raw_data"
    )

    dataset_at = wandb_sdk.wandb_artifacts.Artifact(
        name=f"BRATS2021_{args.data_dim}ClipNorm",
        type="processed_data",
    )

    # check if input folder exists
    if (root / DATA_ROOT / "raw").is_dir():
        raise Exception(f"Folder {root / DATA_ROOT / 'raw'} already exists. Use --force to rerun data processing anyway! (WARNING: this will delete the folder!)")

    print("Downloading dataset version with id: ", args.dataset_version)
    shutil.rmtree(root / DATA_ROOT / "raw")
    data_at.download(root=str(root / DATA_ROOT / "raw"))

    print("Getting filelist from folder")
    processing_list = [
        p for p in (root / DATA_ROOT / "raw").glob("**/*.nii.gz") if p.is_file()
    ]
    OUTPUT_FOLDER.parent.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_FOLDER.is_file() or args.force:
        print("Processing files")
        with h5py.File(str(OUTPUT_FOLDER), "a") as hf:  # open the file in append mode
            for filepath in tqdm(processing_list):
                do_hdf5_processing(filepath, hf)
    else:
        print(
            "Data already processed. You can add --force to rerun data processing anyway!"
        )

    dataset_at.add_file(OUTPUT_FOLDER)
    run.log_artifact(dataset_at)
    wandb_sdk.wandb_run.finish()
