import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from monai.data import NibabelReader
from monai.data.utils import correct_nifti_header_if_necessary
from monai.transforms import Spacingd
from nibabel import Nifti1Image

import wandb


class HDF5NibabelReader(NibabelReader):
    def __init__(self, hdf5_file):
        super().__init__()
        self.hdf5_file_path = str(hdf5_file)

    def verify_suffix(self, filename: Path) -> bool:
        """Verify whether the specified `filename` is supported by the current reader.

        This method should return True if the reader is able to read the
        format suggested by the `filename`.
        """
        return True

    def read(self, data: str, **kwargs) -> Nifti1Image:
        """Read image data from specified file or files. Note that it returns a data
        object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.
        """
        nifti_header = nib.Nifti1Header()
        nifti_affine = None
        # remove tuple if only one file
        data = data[0]

        with h5py.File(str(self.hdf5_file_path), "r") as hdf5_file:
            img = np.array(hdf5_file[data])

            nifti_meta_data = hdf5_file["/".join(data.split("/")[:-1])].attrs
            for key, value in nifti_meta_data.items():
                if "nib.header_" in key:
                    # remove nib.header_ prefix
                    # and set value to nifti header
                    setattr(nifti_header, "_".join(key.split("_")[1:]), value)
            nifti_affine = nifti_meta_data["nib.affine"]

            # convert to nifti
            nib_img3d = nib.Nifti1Image(img, nifti_affine, nifti_header)
            nib_img3d = correct_nifti_header_if_necessary(nib_img3d)

        return nib_img3d
