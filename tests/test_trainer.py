import pytest
import torch
from lightning.pytorch import LightningModule

from pl_cross.ensemble import EnsembleLightningModule
from pl_cross.trainer import KFoldTrainer

from .helper import BoringDataModule, BoringModel, LitClassifier


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ({"num_folds": 2.5}, "Expected argument `num_folds` to be an integer larger than or equal to 2"),
        ({"num_folds": 1}, "Expected argument `num_folds` to be an integer larger than or equal to 2"),
        ({"shuffle": 2}, "Expected argument `shuffle` to be an boolean"),
        ({"stratified": 2}, "Expected argument `stratified` to be an boolean"),
    ],
)
def test_trainer_initialization(arguments, expected):
    """Test additional arguments added to trainer raises error on wrong input."""
    with pytest.raises(ValueError, match=expected):
        KFoldTrainer(**arguments)


@pytest.mark.parametrize("accelerator", ["cpu", "gpu"])
def test_cross_validate(accelerator):
    """Test cross validation finish a basic run."""
    if not torch.cuda.is_available() and torch.cuda.device_count() < 1:
        pytest.skip("test requires cuda support")

    model = BoringModel()
    datamodule = BoringDataModule(feature_size=32)

    trainer = KFoldTrainer(num_folds=2, max_steps=50, accelerator=accelerator, devices=1)
    trainer.cross_validate(model, datamodule=datamodule)


def test_error_on_missing_test_step():
    """Make sure that error is raised if we cannot create an ensemble."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50, devices=1)
    model = LitClassifier()
    datamodule = BoringDataModule(feature_size=32)
    model.test_step = LightningModule.test_step
    with pytest.raises(ValueError, match="`cross_validation` method requires you to also define a `test_step` method."):
        trainer.cross_validate(model, datamodule=datamodule)


def test_ensemble():
    """Test that trainer.create_ensemble works."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50)
    model = LitClassifier()
    datamodule = BoringDataModule(with_labels=True, feature_size=784)

    trainer.cross_validate(model, datamodule=datamodule)
    ensemble_model = trainer.create_ensemble(model)

    assert isinstance(ensemble_model, EnsembleLightningModule)


def test_ensemble_error():
    trainer = KFoldTrainer(num_folds=2, max_steps=50)
    model = LitClassifier()

    with pytest.raises(ValueError, match="Cannot construct ensemble model. Either call `cross_validate`.*"):
        trainer.create_ensemble(model)
