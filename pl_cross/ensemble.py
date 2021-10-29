from pytorch_lightning import LightningModule
import torch
from typing import List, Any

class EnsembleLightningModule(LightningModule):
    def __init__(self, model_cls, checkpoint_paths: List[str]):
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )

    def __getattr__(self, name: str) -> Any:
        attr = super().__getattr__(name)
        if isinstance(attr, callable):
            return [getattr(m, name) for m in range(self.models)]
        return attr
