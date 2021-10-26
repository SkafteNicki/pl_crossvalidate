from abc import ABC, abstractmethod
from pytorch_lightning import LightningDataModule

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
