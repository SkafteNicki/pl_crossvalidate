from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule


class TrainDataLoaderWrapper(LightningDataModule):
    def __init__(self, train_dataloader: DataLoader) -> None:
        super().__init__()
        self._train_dataloader = train_dataloader
        
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self) -> None:
        """ Implement how folds should be initialized """
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        """ 
        Given a fold index, implement how the train and validation
        dataset/dataloader should look for the current fold
        """
        pass
    
@dataclass    
class KFoldDataModule(BaseKFoldDataModule):
    train_fold: Optional[Dataset] = None
    test_fold: Optional[Dataset] = None
    
    def __init__(
        self, 
        num_folds: int = 5,
        shuffle: bool = False,
        stratified: bool = False,
        train_dataloader: Optional[DataLoader] = None,
        datamodule: Optional[LightningDataModule] = None
    ) -> None:
        if train_dataloader is None and datamodule is None:
            raise ValueError('Either `train_dataloader` or `datamodule` argument should be provided')
        if train_dataloader is not None:
            self.datamodule = TrainDataLoaderWrapper(train_dataloader)
        if datamodule is not None:
            self.datamodule = datamodule
        if train_dataloader is not None and datamodule is not None:
            raise ValueError('Only one of `train_dataloader` and `datamodule` argument should be provided')

        self.num_folds = num_folds
        self.shuffle = shuffle
        self.stratified = stratified
        self._dataloader_stats = None
        self.label_extractor = lambda batch: batch[1]  # return second element

    @property
    def dataloader_stats(self):
        """ returns the stats of the train dataloader """
        if self._dataloader_stats is None:
            orig_dl = self.datamodule.train_dataloader()
            self._dataloader_stats = {
                "batch_size": orig_dl.batch_size,
                "num_workers": orig_dl.num_workers,
                "collate_fn": orig_dl.collate_fn,
                "pin_memory": orig_dl.pin_memory,
                "drop_last": orig_dl.drop_last,
                "timeout": orig_dl.timeout,
                "worker_init_fn": orig_dl.worker_init_fn,
                "prefetch_factor": orig_dl.prefetch_factor,
                "persistent_workers": orig_dl.persistent_workers
            }
        return self._dataloader_stats
    
    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)
    
    def setup_folds(self) -> None:
        if self.stratified:
            labels = self.get_train_labels(self.datamodule.train_dataloader())
            if labels is None:
                raise ValueError("Tried to extract labels for stratified K folds but failed."
                                 " Make sure that the dataset of your train dataloader either"
                                 " has an attribute `labels` or that `label_extractor` attribute"
                                 " is initialized correctly")
            splitter = StratifiedKFold(self.num_folds, shuffle=self.shuffle)
        else:
            labels = None
            splitter = KFold(self.num_folds, shuffle=self.shuffle)
        self.train_dataset = self.datamodule.train_dataloader().dataset
        self.splits = [split for split in splitter.split(range(len(self.train_dataset)), y=labels)]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, test_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.test_fold = Subset(self.train_dataset, test_indices)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, **self.dataloader_stats)
    
    def val_dataloader(self) -> DataLoader:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_fold, **self.dataloader_stats)

    def get_train_labels(self, dataloader: DataLoader) -> List:
        if hasattr(dataloader.dataset, "labels"):
            return dataloader.dataset.labels.tolist()
        
        labels = [ ]
        for batch in dataloader:
            try:
                labels.append(self.label_extractor(batch))
            except:
                return None
        labels = torch.cat(labels, dim=0)
        return labels.tolist()


