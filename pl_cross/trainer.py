from typing import Optional

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer as Trainer_pl
from pytorch_lightning.loops.base import Loop

from pl_cross.datamodule import BaseKFoldDataModule, KFoldDataModule


class KFold(Loop):
    def __init__(self):
        pass

class Trainer(Trainer_pl):
    """ 
    Specialized trainer that implements additional methods for easy cross validation
    in pytorch lightning
    
    Args:
        K: number of folds for cross validation
        stratified: boolean indicating if folds should be constructed in a
            stratified way. Currently only supported if you dataset has a `labels`
            attribute.
        *args: additional arguments to pass to normal trainer constructor
        **kwargs: additional keyword arguments to pass to normal trainer constructor
    """
    def __init__(self, K: int = 5, stratified: bool = False, *args, **kwargs):
        self.K = K
        self.stratified = stratified
        super().__init__(*args, **kwargs)

    def cross_validate(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:
        # overwrite standard fit loop
        self.fit_loop = KFold(self.K, self.fit_loop)
        
        # construct K fold datamodule if user is not already passing one in
        cond = (
            train_dataloader is not None or \
            datamodule is not None and not isinstance(datamodule, BaseKFoldDataModule)
        )
        if cond:
            datamodule = KFoldDataModule(train_dataloader, datamodule)
            
        self.fit(model, datamodule=datamodule)
        
        # restore original fit loop
        self.fit_loop = self.fit_loop.fit_loop
