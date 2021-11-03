from typing import Optional
from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer as Trainer_pl
from torch.utils.data import DataLoader

from pl_cross.datamodule import BaseKFoldDataModule, KFoldDataModule
from pl_cross.loop import KFoldLoop


class Trainer(Trainer_pl):
    """
    Specialized trainer that implements additional methods for easy cross validation
    in pytorch lightning

    Args:
        num_folds: number of folds for cross validation
        shuffle: boolean indicating if samples should be shuffled before creating folds
        stratified: boolean indicating if folds should be constructed in a stratified way.
        *args: additional arguments to pass to normal trainer constructor
        **kwargs: additional keyword arguments to pass to normal trainer constructor
    """

    def __init__(
        self, num_folds: int = 5, shuffle: bool = False, stratified: bool = False, *args, **kwargs
    ) -> None:
        if not isinstance(num_folds, int) or num_folds < 2:
            raise ValueError("Expected argument `num_folds` to be an integer larger than or equal to 2")
        self.num_folds = num_folds
        if not isinstance(shuffle, bool):
            raise ValueError("Expected argument `shuffle` to be an boolean")
        self.shuffle = shuffle
        if not isinstance(stratified, bool):
            raise ValueError("Expected argument `stratified` to be an boolean")
        self.stratified = stratified
        super().__init__(*args, **kwargs)

    def cross_validate(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[DataLoader] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:
        # overwrite standard fit loop
        self.fit_loop = KFoldLoop(self.num_folds, self.fit_loop)

        # construct K fold datamodule if user is not already passing one in
        cond = (
            train_dataloader is not None
            or datamodule is not None
            and not isinstance(datamodule, BaseKFoldDataModule)
        )
        if cond:
            datamodule = KFoldDataModule(
                self.num_folds,
                self.shuffle,
                self.stratified,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
            )

        self.fit(model, datamodule=datamodule)

        # restore original fit loop
        self.fit_loop = self.fit_loop.fit_loop

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = super().add_argparse_args(parent_parser, **kwargs)
        parser.add_argument('--num_folds', type=int, default=5)
        parser.add_argument('--shuffle', type=bool, default=False)
        parser.add_argument('--stratified', type=bool, default=False)
        return parser



