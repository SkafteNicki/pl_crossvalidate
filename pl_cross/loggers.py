from pytorch_lightning import loggers as pl_loggers

class KFoldLogger:
    def setup(self):
        """ Additional setup code to inject during __init__ """
        self._fold_idx = 0
        self._version = f"fold{self._fold_idx}"

    def increment(self):
        """ Will run after an fold has been executed """
        self._experiment = None
        self._fold_idx += 1
        self._version = f"fold{self._fold_idx}"


class CometLogger(pl_loggers.CometLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()


class CSVLogger(pl_loggers.CSVLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()


class NeptuneLogger(pl_loggers.MLFlowLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()


class TensorboardLogger(pl_loggers.TensorBoardLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()


class TestTubeLogger(pl_loggers.TestTubeLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()


class WandbLogger(pl_loggers.WandbLogger, KFoldLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()
