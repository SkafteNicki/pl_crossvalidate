import pytest

from pl_cross.trainer import EnsembleLightningModule, Trainer

from .boring_model import BoringDataModule, BoringModel, LitClassifier

_paths = [f"tests/ensemble_weights/model_fold{i}.pt" for i in range(5)]
_n_ensemble = len(_paths)


def test_trainer_initialization():
    """ Test additional arguments added to trainer """
    with pytest.raises(
        ValueError, match="Expected argument `num_folds` to be an integer larger than or equal to 2"
    ):
        Trainer(num_folds=2.5)

    with pytest.raises(
        ValueError, match="Expected argument `num_folds` to be an integer larger than or equal to 2"
    ):
        Trainer(num_folds=1)

    with pytest.raises(ValueError, match="Expected argument `shuffle` to be an boolean"):
        Trainer(shuffle=2)

    with pytest.raises(ValueError, match="Expected argument `stratified` to be an boolean"):
        Trainer(stratified=2)


@pytest.mark.parametrize("accelerator", ["cpu", "gpu"])
def test_cross_validate(accelerator):
    """ test cross validation works """
    model = BoringModel()
    datamodule = BoringDataModule()

    trainer = Trainer(num_folds=2, max_steps=50, accelerator=accelerator, devices=1)
    trainer.cross_validate(model, datamodule=datamodule)


@pytest.mark.parametrize("paths", [None, _paths])
def test_ensemble(paths):
    """ test that trainer.ensemble works with and without that paths argument """
    trainer = Trainer(num_folds=_n_ensemble, max_steps=50)
    model = BoringModel()
    datamodule = BoringDataModule()

    if paths:
        ensemble_model = trainer.create_ensemble(model, paths)
    else:
        trainer.cross_validate(model, datamodule=datamodule)
        ensemble_model = trainer.create_ensemble(model)

    assert isinstance(ensemble_model, EnsembleLightningModule)


def test_ensemble_error():
    """ make sure that error is raised if we cannot create an ensemble """
    trainer = Trainer(num_folds=_n_ensemble)
    model = LitClassifier()
    with pytest.raises(ValueError, match="Cannot construct.*"):
        trainer.create_ensemble(model)
