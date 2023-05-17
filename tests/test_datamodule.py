import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pl_crossvalidate import KFoldDataModule

from . import BoringDataModule, RandomDataset, RandomLabelDataset

train_dataloader = DataLoader(RandomDataset(32, 64))
datamodule = BoringDataModule()


@pytest.mark.parametrize(
    "input, expected",
    [
        ((5, False, False), "Either `train_dataloader` or `datamodule` argument should be provided"),
        (
            (5, False, False, train_dataloader, None, datamodule),
            "Only one of `train_dataloader` and `datamodule` argument should be provided",
        ),
        ((1, False, False, train_dataloader), "Number of folds must be a positive integer larger than 2"),
        (
            (5, 2, False, train_dataloader),
            "Shuffle must be a boolean value",
        ),
        ((5, False, 2, train_dataloader), "Stratified must be a boolean value"),
    ],
)
def test_initialization_errors(input, expected):
    """Test that correct errors are raised when initializing KFoldDataModule."""
    with pytest.raises(ValueError, match=expected):
        KFoldDataModule(*input)


def test_initialization():
    """Test that the underlying datamodule/loader does not change after init."""
    datamodule = KFoldDataModule(5, False, train_dataloader=train_dataloader, val_dataloaders=train_dataloader)
    assert datamodule.datamodule.train_dataloader() == train_dataloader
    assert datamodule.datamodule.val_dataloader() == train_dataloader
    assert datamodule.val_dataloader() == train_dataloader

    datamodule = KFoldDataModule(5, False, datamodule=datamodule)
    assert datamodule.datamodule == datamodule


def test_getting_stats():
    """Test that the underlying attributes of the dataloader are the same as the original dataloader."""
    train_dataloader = DataLoader(RandomDataset(32, 64))
    datamodule = KFoldDataModule(5, False, False, train_dataloader=train_dataloader)

    new_train_dataloader = datamodule.train_dataloader()
    for attr in [
        "batch_size",
        "num_workers",
        "collate_fn",
        "pin_memory",
        "drop_last",
        "timeout",
        "worker_init_fn",
        "prefetch_factor",
        "persistent_workers",
    ]:
        assert getattr(train_dataloader, attr) == getattr(new_train_dataloader, attr)


@pytest.mark.parametrize("num_folds", [2, 5, 10])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize(
    "data, module_type",
    [(DataLoader(RandomDataset(32, 64)), "train_dataloader"), (BoringDataModule(), "datamodule")],
)
def test_getting_folds(num_folds, shuffle, data, module_type):
    """Test that the folds are correctly generated."""
    kwargs = {module_type: data}
    datamodule = KFoldDataModule(num_folds, shuffle, False, **kwargs)

    for fold_index in range(num_folds):
        datamodule.fold_index = fold_index
        train_dl = datamodule.train_dataloader()
        test_dl = datamodule.test_dataloader()
        assert isinstance(train_dl, DataLoader)
        assert isinstance(test_dl, DataLoader)

        # make sure no element were lost
        train_data = torch.cat([batch for batch in train_dl], dim=0)
        test_data = torch.cat([batch for batch in test_dl], dim=0)
        assert len(train_data) + len(test_data) == len(datamodule.datamodule.train_dataloader().dataset)

        # make sure elements to not overlap (assuming all elements are unique)
        for td in train_data:
            assert td not in test_data


@pytest.mark.parametrize(
    "data, module_type, add_labels",
    [
        (DataLoader(RandomLabelDataset(32, 64)), "train_dataloader", False),
        (BoringDataModule(with_labels=True), "datamodule", False),
        (DataLoader(RandomDataset(32, 64)), "train_dataloader", True),
        (BoringDataModule(with_labels=False), "datamodule", True),
    ],
)
def test_stratified(data, module_type, add_labels):
    """Test that stratified splitting works as expected."""
    kwargs = {module_type: data}
    stratified = KFoldDataModule(num_folds=5, shuffle=False, stratified=True, **kwargs)
    not_stratified = KFoldDataModule(num_folds=5, shuffle=False, stratified=False, **kwargs)

    # if we add the labels attribute to our dataset we should still
    # be able to do stratified splitting
    if add_labels:
        stratified.datamodule.train_dataloader().dataset.labels = torch.randint(
            3, (len(stratified.datamodule.train_dataloader().dataset),)
        )

    for dm in [stratified, not_stratified]:
        dm.setup_folds()  # initialize the splits

    # make sure that by using stratified splitting, the splits actually change
    equal = True
    for split1, split2 in zip(stratified.splits, not_stratified.splits, strict=True):
        equal = equal and np.allclose(split1[0], split2[0])
        equal = equal and np.allclose(split1[1], split2[1])
    assert not equal


@pytest.mark.parametrize("custom", [False, True])
def test_custom_stratified_label_extractor(custom):
    """Test that error is raised if the custom label extractor does not work and that everything work if it does."""
    train_dataloader = DataLoader(RandomDataset(32, 64))
    stratified = KFoldDataModule(stratified=True, train_dataloader=train_dataloader)
    if custom:
        stratified.label_extractor = lambda batch: batch["b"]

    if custom:
        # should work
        assert hasattr(stratified, "splits")
    else:
        with pytest.raises(ValueError, match="Tried to extract labels for .*"):
            # should not work, label extraction should fail
            stratified.train_dataloader()


def test_get_labels():
    """Test that the labels are correctly extracted."""
    train_dataloader = DataLoader(RandomLabelDataset(32, 64))
    stratified = KFoldDataModule(stratified=True, train_dataloader=train_dataloader, val_dataloaders=train_dataloader)

    for dl in [stratified.datamodule.train_dataloader(), stratified.datamodule.val_dataloader()]:
        labels = stratified.get_labels(dl)
        assert len(labels) == len(dl.dataset)
        assert isinstance(labels, list)
