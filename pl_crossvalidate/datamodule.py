from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset


class DataloaderToDataModule(LightningDataModule):
    """Converts a set of dataloaders into a lightning datamodule.

    Args:
        train_dataloader: Training dataloader
        val_dataloaders: Validation dataloader(s)
        test_dataloaders: Test dataloader(s)
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloaders: Union[DataLoader, Sequence[DataLoader]],
    ) -> None:
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloaders = val_dataloaders

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self._train_dataloader

    def val_dataloader(self) -> Union[DataLoader, Sequence[DataLoader]]:
        """Return validation dataloader(s)."""
        return self._val_dataloaders


@dataclass
class KFoldDataModule(LightningDataModule):
    """K-Fold cross validation datamodule.

    Specialized datamodule that can be used for K-Fold cross validation. The first time the `train_dataloader` or
    `test_dataloader` method is call, K folds are generated and the dataloaders are created based on the current fold.

    The input is either a single training dataloader (with an optional validation dataloader) or a lightning datamodule
    that then gets wrapped.

    Args:
        num_folds: Number of folds
        shuffle: Whether to shuffle the data before splitting it into folds
        stratified: Whether to use stratified sampling e.g. for classification we make sure that each fold has the same
            ratio of samples from each class as the original dataset
        train_dataloader: Training dataloader
        val_dataloaders: Validation dataloader(s)
        datamodule: Lightning datamodule

    """

    def __init__(
        self,
        num_folds: int = 5,
        shuffle: bool = False,
        stratified: bool = False,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, Sequence[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        super().__init__()
        # Input validation
        if train_dataloader is None and datamodule is None:
            raise ValueError("Either `train_dataloader` or `datamodule` argument should be provided")
        if train_dataloader is not None:
            self.datamodule = DataloaderToDataModule(train_dataloader, val_dataloaders)
        if datamodule is not None:
            self.datamodule = datamodule
        if train_dataloader is not None and datamodule is not None:
            raise ValueError("Only one of `train_dataloader` and `datamodule` argument should be provided")

        if not (isinstance(num_folds, int) and num_folds >= 2):
            raise ValueError("Number of folds must be a positive integer larger than 2")
        self.num_folds = num_folds

        if not isinstance(shuffle, bool):
            raise ValueError("Shuffle must be a boolean value")
        self.shuffle = shuffle

        if not isinstance(stratified, bool):
            raise ValueError("Stratified must be a boolean value")
        self.stratified = stratified

        self.fold_index = 0
        self.splits = None
        self.dataloader_settings = None
        self.label_extractor = lambda batch: batch[1]  # return second element

    def setup_folds(self) -> None:
        """Implement how folds should be initialized."""
        if self.splits is None:
            labels = None
            if self.stratified:
                labels = self.get_labels(self.datamodule.train_dataloader())
                if labels is None:
                    raise ValueError(
                        "Tried to extract labels for stratified K folds but failed."
                        " Make sure that the dataset of your train dataloader either"
                        " has an attribute `labels` or that `label_extractor` attribute"
                        " is initialized correctly"
                    )
                splitter = StratifiedKFold(self.num_folds, shuffle=self.shuffle)
            else:
                splitter = KFold(self.num_folds, shuffle=self.shuffle)
            self.train_dataset = self.datamodule.train_dataloader().dataset
            self.splits = [split for split in splitter.split(range(len(self.train_dataset)), y=labels)]

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader on the current fold."""
        self.setup_folds()
        train_fold = Subset(self.train_dataset, self.splits[self.fold_index][0])
        return DataLoader(train_fold, **self.dataloader_setting)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader, which is the same regardless of the fold."""
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader on the current fold."""
        self.setup_folds()
        test_fold = Subset(self.train_dataset, self.splits[self.fold_index][1])
        return DataLoader(test_fold, **self.dataloader_setting)

    def get_labels(self, dataloader: DataLoader) -> Optional[List]:
        """Try to extract the training labels (for classification problems) from the underlying training dataset."""
        # Try to extract labels from the dataset through labels attribute
        if hasattr(dataloader.dataset, "labels"):
            return dataloader.dataset.labels.tolist()

        # Else iterate and try to extract
        try:
            return torch.cat([self.label_extractor(batch) for batch in dataloader], dim=0).tolist()
        except Exception:
            return None

    @property
    def dataloader_setting(self) -> dict:
        """Return the settings of the train dataloader."""
        if self.dataloader_settings is None:
            orig_dl = self.datamodule.train_dataloader()
            self.dataloader_settings = {
                "batch_size": orig_dl.batch_size,
                "num_workers": orig_dl.num_workers,
                "collate_fn": orig_dl.collate_fn,
                "pin_memory": orig_dl.pin_memory,
                "drop_last": orig_dl.drop_last,
                "timeout": orig_dl.timeout,
                "worker_init_fn": orig_dl.worker_init_fn,
                "prefetch_factor": orig_dl.prefetch_factor,
                "persistent_workers": orig_dl.persistent_workers,
            }
        return self.dataloader_settings
