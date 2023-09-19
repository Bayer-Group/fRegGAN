import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BRATS2021(Dataset):
    def __init__(
        self,
        hdf5_file: Path,
        datasplit_file: Path,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        final_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.target_transform = target_transform
        self.final_transform = final_transform
        self.train = train  # training set or test set

        self.data, self.targets = self._load_data(datasplit_file)

    def _load_data(self, datasplit_file):
        file_data = pd.read_csv(str(datasplit_file), sep=" ", header=None).to_numpy()
        inputs = list(file_data[:, 0])
        targets = list(file_data[:, 1])
        return inputs, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a image where we want to translat to.
        """
        with h5py.File(str(self.hdf5_file), "r") as hdf5_file:
            img, target = np.array(hdf5_file[self.data[index]]), np.array(
                hdf5_file[self.targets[index]]
            )

        if self.transform is not None:
            transformed = self.transform(image=img, target=target)
            img = transformed["image"]
            target = transformed["target"]

        if self.target_transform is not None:
            target = self.target_transform(image=target)["image"]

        if self.final_transform is not None:
            transformed = self.final_transform(image=img, target=target)
            img = transformed["image"]
            target = transformed["target"]

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Val"
        return f"Split: {split}"


class BRATS2021Unpaired(BRATS2021):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a image where we want to translat to.
        """
        with h5py.File(str(self.hdf5_file), "r") as hdf5_file:
            # data will be unpaired for training
            if self.train:
                target_index = random.choice(
                    list(set(range(0, len(self.targets))).difference({index}))
                )
            else:
                target_index = index
            img, target = np.array(hdf5_file[self.data[index]]), np.array(
                hdf5_file[self.targets[target_index]]
            )

        if self.transform is not None:
            transformed = self.transform(image=img, target=target)
            img = transformed["image"]
            target = transformed["target"]

        if self.target_transform is not None:
            target = self.target_transform(image=target)["image"]

        if self.final_transform is not None:
            transformed = self.final_transform(image=img, target=target)
            img = transformed["image"]
            target = transformed["target"]

        return img, target
