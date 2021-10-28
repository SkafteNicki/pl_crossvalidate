import pytest

from .boring_model import RandomDataset, BoringDataModule, RandomLabelDataset, RandomDictLabelDataset
from pl_cross import KFoldDataModule

import torch
from torch.utils.data import DataLoader

def test_initialization():
    train_dataloader = DataLoader(RandomDataset(32, 64))
    datamodule = BoringDataModule()
    
    with pytest.raises(ValueError):
        KFoldDataModule(5, False, False)
        
    with pytest.raises(ValueError):
        KFoldDataModule(5, False, False, train_dataloader, datamodule)
        
    datamodule = KFoldDataModule(5, False, train_dataloader=train_dataloader)
    assert datamodule.datamodule.train_dataloader() == train_dataloader
    
    datamodule = KFoldDataModule(5, False, datamodule=datamodule)
    assert datamodule.datamodule == datamodule


def test_getting_stats():
    train_dataloader = DataLoader(RandomDataset(32, 64))
    
    datamodule = KFoldDataModule(5, False, False, train_dataloader=train_dataloader)
    datamodule.setup_folds()

    new_train_dataloader = datamodule.train_dataloader()

    assert train_dataloader.batch_size == new_train_dataloader.batch_size

@pytest.mark.parametrize("num_folds", [2, 5, 10])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("data, dtype", [
    (DataLoader(RandomDataset(32, 64)), "train_dataloader"), 
    (BoringDataModule(), "datamodule")
])
def test_getting_folds(num_folds, shuffle, data, dtype):
    kwargs = {dtype: data}
    datamodule = KFoldDataModule(num_folds, shuffle, False, **kwargs)
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.setup_folds()

    for fold_idx in range(num_folds):
        datamodule.setup_fold_index(fold_idx)
        train_dl = datamodule.train_dataloader()
        test_dl = datamodule.test_dataloader()
        assert train_dl
        assert test_dl

        # make sure no element were lost
        train_data = torch.cat([batch for batch in train_dl], dim=0)
        test_data = torch.cat([batch for batch in test_dl], dim=0)
        assert len(train_data) + len(test_data) == len(datamodule.datamodule.train_dataloader().dataset)

        # make sure elements to not overlap (assuming all elements are unique)
        for td in train_data:
            assert td not in test_data


@pytest.mark.parametrize("data, dtype, add_labels", [
    (DataLoader(RandomLabelDataset(32, 64)), "train_dataloader", False), 
    (BoringDataModule(with_labels=True), "datamodule", False),
    (DataLoader(RandomDataset(32, 64)), "train_dataloader", True), 
    (BoringDataModule(with_labels=False), "datamodule", True)
])
def test_stratified(data, dtype, add_labels):
    kwargs = {dtype: data}
    stratified = KFoldDataModule(num_folds=5, shuffle=False, stratified=True, **kwargs)
    not_stratified = KFoldDataModule(num_folds=5, shuffle=False, stratified=False, **kwargs)

    for dm in [stratified, not_stratified]:
        dm.prepare_data()
        dm.setup()

    # if we add the labels attribute to our dataset we should still 
    # be able to do stratified splitting
    if add_labels:
        stratified.datamodule.train_dataloader().dataset.labels = \
            torch.randint(2, (len(stratified.datamodule.train_dataloader().dataset),))

    for dm in [stratified, not_stratified]:
        dm.setup_folds()

    # make sure that by using stratified splitting, the splits actually change
    assert stratified.splits != not_stratified


@pytest.mark.parametrize("custom", [False, True])
def test_custom_stratified_label_extractor(custom):
    train_dataloader = DataLoader(RandomDictLabelDataset(32, 64))
    stratified = KFoldDataModule(stratified=True, train_dataloader=train_dataloader)
    if custom:
        stratified.label_extractor = lambda batch: batch["b"]
    
    stratified.prepare_data()
    stratified.setup()

    if custom:
        # should work
        stratified.setup_folds()
    else:
        with pytest.raises(ValueError, match="Tried to extract labels for .*"):
            # should not work, label extraction should fail
            stratified.setup_folds()


@pytest.mark.parametrize("data, dtype", [
    (DataLoader(RandomDataset(32, 64)), "train_dataloader"), 
    (BoringDataModule(with_labels=False), "datamodule")
])
def test_error_stratified(data, dtype):
    kwargs = {dtype: data}
    stratified = KFoldDataModule(num_folds=5, shuffle=False, stratified=True, **kwargs)

    stratified.prepare_data()
    stratified.setup()

    with pytest.raises(ValueError):
        stratified.setup_folds()