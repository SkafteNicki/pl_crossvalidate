from pytorch_lightning import LightningModule
import torch
from torch import nn, Tensor
from typing import List, Any, Optional, Callable
import functools
import inspect


class EnsembleLightningModule(LightningModule):
    """ EnsembleLightningModule can be used to constuct an ensemble
    from a collection of models. Under the hood we wrap every public method defined in the
    model to return the output from all models in the ensemble. If the output is a tensor,
    we stack them before returning.

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
    exclude = [
        'extra_repr',
        'loaded_optimizer_states_dict',  # TODO: remove when pl==1.7.0
        'model_size'  # TODO: remove when pl==1.7.0
    ]

    def __init__(self, model: LightningModule, ckpt_paths: List[str]) -> None:
        super().__init__()
        model_cls = type(model)
        self.models = nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in ckpt_paths]
        )

        for attr in inspect.getmembers(model, inspect.ismethod):
            attr_name = attr[0]
            if not attr_name.startswith("_") and attr_name not in self.exclude:
                setattr(self, attr_name, self.wrap_callables(getattr(self, attr_name)))

    def wrap_callables(self, fn: Callable) -> Callable:
        """ Decorato to wrap a function method to return the collected output from 
        all models in the ensemble.
        """
        @functools.wraps(fn)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            val = [getattr(m, fn.__name__)(*args, **kwargs) for m in self.models]
            if isinstance(val, Tensor):
                val = torch.stack(val)
            return val

        return wrapped_func
