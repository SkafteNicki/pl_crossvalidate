import pytest

from .boring_model import RandomDataset, BoringDataModule
from pl_cross import KFoldDataModule

from torch.utils.data import DataLoader

def test_initialization():
    train_dataloader = DataLoader(RandomDataset(32, 64))
    datamodule = BoringDataModule()
    
    with pytest.raises(ValueError):
        KFoldDataModule(5, False)
        
    with pytest.raises(ValueError):
        KFoldDataModule(5, False, train_dataloader, datamodule)
        
    datamodule = KFoldDataModule(5, False, train_dataloader=train_dataloader)
    assert datamodule.datamodule.train_dataloader() == train_dataloader
    
    datamodule = KFoldDataModule(5, False, datamodule=datamodule)
    assert datamodule.datamodule == datamodule
    