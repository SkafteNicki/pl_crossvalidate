import pytest
import torch
from torch import Tensor

from pl_cross import EnsembleLightningModule, KFoldTrainer

from .helper import BoringDataModule, BoringModel, LitClassifier


@pytest.fixture(scope="module")
def paths():
    """Create paths."""
    trainer = KFoldTrainer(num_folds=2, max_steps=50)
    model = LitClassifier()
    datamodule = BoringDataModule(with_labels=True, feature_size=784)
    trainer.cross_validate(model, datamodule=datamodule)
    return trainer._ensemple_paths


def test_init_emsemble(paths):
    """Check that we can initialize an ensemble."""
    model = LitClassifier()
    emodel = EnsembleLightningModule(model, paths)
    for m in emodel.models:
        assert isinstance(m, LitClassifier)


def test_init_with_correct_model_class(paths):
    """Check that an error is raised if we try to initialize with wrong model class."""
    model = BoringModel()
    with pytest.raises(RuntimeError, match=".*in loading state_dict.*"):
        EnsembleLightningModule(model, paths)


def test_callable_methods(paths):
    """Check that methods from base model can be called from ensemble."""
    model = LitClassifier()
    emodel = EnsembleLightningModule(model, paths)
    for m in emodel.models:
        assert isinstance(m, LitClassifier)

    # Forward returns tensors, so output should be single stacked tensor
    output = emodel(torch.randn(5, 784))
    assert isinstance(output, Tensor)
    assert output.shape[0] == len(paths)

    # configure_optimizers returns non-tensors, so should just be a list
    output = emodel.configure_optimizers()
    assert isinstance(output, list)
    assert len(output) == len(paths)
