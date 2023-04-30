from copy import deepcopy

import pytest
import torch
from lightning.pytorch import LightningModule

from pl_cross.datamodule import KFoldDataModule
from pl_cross.ensemble import EnsembleLightningModule
from pl_cross.trainer import KFoldTrainer

from . import BoringDataModule, BoringModel, LitClassifier


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


def test_cross_validate_gets_correctly_reset_between_runs():
    """Test that the trainer is correctly reset between runs."""

    class CheckBetweenRuns(BoringModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._initial_state = deepcopy(self.state_dict())

        def on_train_start(self):
            state_dict = self.state_dict()
            for key in state_dict:
                assert torch.allclose(state_dict[key], self._initial_state[key])

    model = CheckBetweenRuns()
    datamodule = BoringDataModule(feature_size=32)

    trainer = KFoldTrainer(num_folds=2, max_steps=50, accelerator="cpu")
    trainer.cross_validate(model, datamodule=datamodule)


def test_error_on_missing_test_step():
    """Make sure that error is raised if we cannot create an ensemble."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50, devices=1)
    model = LitClassifier()
    datamodule = BoringDataModule(feature_size=32)
    model.test_step = LightningModule.test_step
    with pytest.raises(ValueError, match="`cross_validation` method requires you to also define a `test_step` method."):
        trainer.cross_validate(model, datamodule=datamodule)


@pytest.fixture(scope="module")
def paths():
    """Create paths."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50)
    model = LitClassifier()
    datamodule = BoringDataModule(with_labels=True, feature_size=784)
    trainer.cross_validate(model, datamodule=datamodule)
    return trainer._ensemple_paths


@pytest.mark.parametrize("ckpt_paths", [None, "paths"])
def test_ensemble(ckpt_paths, request):
    """Test that trainer.create_ensemble works."""
    if isinstance(ckpt_paths, str):
        ckpt_paths = request.getfixturevalue(ckpt_paths)

    trainer = KFoldTrainer(num_folds=2, max_steps=50)
    model = LitClassifier()
    datamodule = BoringDataModule(with_labels=True, feature_size=784)

    if ckpt_paths is None:
        trainer.cross_validate(model, datamodule=datamodule)
    ensemble_model = trainer.create_ensemble(model, ckpt_paths=ckpt_paths)

    assert isinstance(ensemble_model, EnsembleLightningModule)


def test_ensemble_error():
    """Test that an error is raised if we try to create an ensemble without calling `cross_validate` first."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50)
    model = LitClassifier()

    with pytest.raises(ValueError, match="Cannot construct ensemble model. Either call `cross_validate`.*"):
        trainer.create_ensemble(model)


def test_out_of_sample_missing_score_method():
    """Test that an error is raised if we try to call `out_of_sample_score` without defining a `score` method."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50)
    model = LitClassifier()

    with pytest.raises(ValueError, match="`out_of_sample_score` method requires you to also define a `score` method."):
        trainer.out_of_sample_score(model)


@pytest.mark.parametrize("ckpt_paths", [None, "paths"])
def test_out_of_sample_method_config_errors(ckpt_paths, request):
    if isinstance(ckpt_paths, str):
        ckpt_paths = request.getfixturevalue(ckpt_paths)

    trainer = KFoldTrainer(num_folds=2, max_steps=50)

    class LitClassifierWithScore(LitClassifier):
        def score(self, batch, batch_idx):
            return self(batch[0]).softmax(dim=-1)

    model = LitClassifierWithScore()

    datamodule = BoringDataModule(with_labels=True, feature_size=784)
    k_datamodule = KFoldDataModule(num_folds=5, datamodule=datamodule)

    if ckpt_paths is None:
        with pytest.raises(ValueError, match="Cannot construct ensemble model. Either call `cross_validate`"):
            trainer.out_of_sample_score(model)
    else:
        with pytest.raises(ValueError, match="Cannot compute out of sample scores. Either call `cross_validate`.*"):
            trainer.out_of_sample_score(model, ckpt_paths=ckpt_paths)

        with pytest.raises(ValueError, match="`datamodule` argument must be an instance of `KFoldDataModule`."):
            trainer.out_of_sample_score(model, datamodule=datamodule, ckpt_paths=ckpt_paths)

        with pytest.raises(
            ValueError, match="Number of checkpoint paths provided does not match the number of folds in the datamodule"
        ):
            trainer.out_of_sample_score(model, datamodule=k_datamodule, ckpt_paths=ckpt_paths)


@pytest.mark.parametrize("shuffle", [False, True])
def test_out_sample_method_correctness(shuffle):
    trainer = KFoldTrainer(num_folds=2, shuffle=shuffle, max_steps=50)

    dataset = torch.utils.data.TensorDataset(
        torch.arange(110).unsqueeze(1).repeat(1, 784).float(), torch.randint(2, (110,))
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=shuffle)

    class LitClassifierWithScore(LitClassifier):
        def score(self, batch, batch_idx):
            return batch[0]  # return labels which we can check against

    model = LitClassifierWithScore()

    trainer.cross_validate(model, train_dataloader=dataloader, val_dataloaders=dataloader)

    out = trainer.out_of_sample_score(model)
    for i, o in enumerate(out):
        assert o.sum() == i * 784, "out of sample score is incorrectly shuffled"
