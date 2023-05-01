import os
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
def test_cross_validate(tmp_path, accelerator):
    """Test cross validation finish a basic run."""
    if not torch.cuda.is_available() and torch.cuda.device_count() < 1:
        pytest.skip("test requires cuda support")

    model = BoringModel()
    datamodule = BoringDataModule(feature_size=32)

    trainer = KFoldTrainer(num_folds=2, max_steps=50, accelerator=accelerator, devices=1, default_root_dir=tmp_path)
    trainer.cross_validate(model, datamodule=datamodule)


def test_cross_validate_gets_correctly_reset_between_runs(tmp_path):
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

    trainer = KFoldTrainer(num_folds=2, max_steps=50, accelerator="cpu", default_root_dir=tmp_path)
    trainer.cross_validate(model, datamodule=datamodule)


def test_error_on_missing_test_step(tmp_path):
    """Make sure that error is raised if we cannot create an ensemble."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50, devices=1, default_root_dir=tmp_path)
    model = LitClassifier()
    datamodule = BoringDataModule(feature_size=32)
    model.test_step = LightningModule.test_step
    with pytest.raises(ValueError, match="`cross_validation` method requires you to also define a `test_step` method."):
        trainer.cross_validate(model, datamodule=datamodule)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    """Create paths."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50, default_root_dir=tmp_path_factory.mktemp("paths"))
    model = LitClassifier()
    datamodule = BoringDataModule(with_labels=True, feature_size=784)
    trainer.cross_validate(model, datamodule=datamodule)
    return trainer._ensemple_paths


@pytest.mark.parametrize("ckpt_paths", [None, "paths"])
def test_ensemble(tmp_path, ckpt_paths, request):
    """Test that trainer.create_ensemble works."""
    if isinstance(ckpt_paths, str):
        ckpt_paths = request.getfixturevalue(ckpt_paths)

    trainer = KFoldTrainer(num_folds=2, max_steps=50, default_root_dir=tmp_path)
    model = LitClassifier()
    datamodule = BoringDataModule(with_labels=True, feature_size=784)

    if ckpt_paths is None:
        trainer.cross_validate(model, datamodule=datamodule)
    ensemble_model = trainer.create_ensemble(model, ckpt_paths=ckpt_paths)

    assert isinstance(ensemble_model, EnsembleLightningModule)


def test_ensemble_error(tmp_path):
    """Test that an error is raised if we try to create an ensemble without calling `cross_validate` first."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50, default_root_dir=tmp_path)
    model = LitClassifier()

    with pytest.raises(ValueError, match="Cannot construct ensemble model. Either call `cross_validate`.*"):
        trainer.create_ensemble(model)


def test_out_of_sample_missing_score_method(tmp_path):
    """Test that an error is raised if we try to call `out_of_sample_score` without defining a `score` method."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50, default_root_dir=tmp_path)
    model = LitClassifier()

    with pytest.raises(ValueError, match="`out_of_sample_score` method requires you to also define a `score` method."):
        trainer.out_of_sample_score(model)


@pytest.mark.parametrize("ckpt_paths", [None, "paths"])
def test_out_of_sample_method_config_errors(tmp_path, ckpt_paths, request):
    """Test that appropriate errors are raised when calling `out_of_sample_score`."""
    if isinstance(ckpt_paths, str):
        ckpt_paths = request.getfixturevalue(ckpt_paths)

    trainer = KFoldTrainer(num_folds=2, max_steps=50, default_root_dir=tmp_path)

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
def test_out_sample_method_correctness(tmp_path, shuffle):
    """Test that out of sample scores are correctly computed even if the dataloader is shuffled."""
    trainer = KFoldTrainer(num_folds=2, shuffle=shuffle, max_steps=50, default_root_dir=tmp_path)

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


def test_trainer_default_save_structure(tmp_path):
    """Test that the default save structure is correct."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50, default_root_dir=tmp_path)
    model = LitClassifier()
    datamodule = BoringDataModule(with_labels=True, feature_size=784)
    trainer.cross_validate(model, datamodule=datamodule)

    for val in ["fold_0", "fold_1", "kfold_initial_weights.ckpt"]:
        assert val in os.listdir(str(tmp_path) + "/lightning_logs/version_0")
    for i in range(2):
        for val in [f"fold_{i}.ckpt", "hparams.yaml", "metrics.csv"]:
            assert val in os.listdir(str(tmp_path) + f"/lightning_logs/version_0/fold_{i}")
