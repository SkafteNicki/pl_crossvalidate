import inspect
from typing import Any, Callable, List, Optional

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn


class EnsembleLightningModule(LightningModule):
    """EnsembleLightningModule can be used to constuct an ensemble from a collection of models.

    Under the hood we wrap every public method defined in the model to return the output from all models in the
    ensemble. If the output is a tensor, we stack them before returning.

    Args:
        model: A instance of the model you want to turn into an ensemble
        ckpt_paths: A list of strings with paths to checkpoints for the given

    Example:
        >>> model = MyLitModel(...)
        >>> ensemble_model = EnsembleLightningModule(
        ...   model, ['ckpt/path/model1.ckpt', 'ckpt/path/model2.ckpt']
        ... )
        >>> ensemble.predict(...)  # call whatever method your model contain
        [output_from_model1, output_from_model2]

    """

    _allowed_methods = None

    def __init__(self, model: LightningModule, ckpt_paths: List[str]) -> None:
        super().__init__()
        self.models = nn.ModuleList([type(model).load_from_checkpoint(p) for p in ckpt_paths])

        # We need to set the trainer to something to avoid errors
        model._trainer = object()
        self._allowed_methods = [
            attr[0] for attr in inspect.getmembers(model, inspect.ismethod) if not attr[0].startswith("_")
        ]
        model._trainer = None

    def __getattribute__(self, name: str) -> Any:
        """Overwrite default behavior such that the ensemble has the same public methods as the base model."""
        if name == "_allowed_methods":  # break recursion
            return super().__getattribute__(name)
        if self._allowed_methods is not None and name in self._allowed_methods:
            return self.wrap_callables(name)
        return super().__getattribute__(name)

    def wrap_callables(self, name: str) -> Callable:
        """Decorato to wrap a function method to return the collected output from all models in the ensemble."""

        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            val = [getattr(m, name)(*args, **kwargs) for m in self.models]
            if isinstance(val[0], Tensor):
                val = torch.stack(val)
            return val

        return wrapped_func
