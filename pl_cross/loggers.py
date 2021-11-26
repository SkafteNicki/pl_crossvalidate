from pytorch_lightning import loggers as pl_loggers

class KFoldLogger:
    def setup(self):
        self._fold_idx = 0
        self._version = f"fold{self._fold_idx}"

    def increment(self):
        self._experiment = None
        self._fold_idx += 1
        self._version = f"fold{self._fold_idx}"

class TensorboardLogger(pl_loggers.TensorBoardLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()

class CSVLogger(pl_loggers.CSVLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()

class WandbLogger(pl_loggers.WandbLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()
