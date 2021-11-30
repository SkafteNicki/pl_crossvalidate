import pytest
import torch
from torch import Tensor

from pl_cross import EnsembleLightningModule

from .boring_model import BoringModel, LitClassifier

_paths = [f"tests/ensemble_weights/model_fold{i}.pt" for i in range(5)]
_n_ensemble = len(_paths)


def test_ensemble_with_correct_model_class():
    """ Check that we can initialize an ensemble """
    model = LitClassifier()
    emodel = EnsembleLightningModule(model, _paths)
    for m in emodel.models:
        assert isinstance(m, LitClassifier)


def test_init_with_correct_model_class():
    model = BoringModel()
    with pytest.raises(RuntimeError, match=".*in loading state_dict.*"):
        emodel = EnsembleLightningModule(model, _paths)


def test_callable_methods():
    """ Check that methods from base model can be called from ensemble """
    model = LitClassifier()
    emodel = EnsembleLightningModule(model, _paths)
    for m in emodel.models:
        assert isinstance(m, LitClassifier)

    # Forward returns tensors, so output should be single stacked tensor
    output = emodel(torch.randn(5, 784))
    assert isinstance(output, Tensor)
    assert output.shape[0] == _n_ensemble

    # configure_optimizers returns non-tensors, so should just be a list
    output = emodel.configure_optimizers()
    assert isinstance(output, list)
    assert len(output) == _n_ensemble
