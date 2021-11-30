import logging
import os.path as osp
from copy import deepcopy
from typing import Any, Dict

import torch
from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_info

from pl_cross.datamodule import BaseKFoldDataModule

_logger = logging.getLogger('pl_cross')

class KFoldLoop(Loop):
    """ Specialized pytorch lightning loop for doing cross validation 
    Adjusted version of:
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py

    Args:
        num_folds: number of K folds to do
        fit_loop: base fit loop from a instance of pytorch_lightning.Trainer

    """
    def __init__(self, num_folds: int, fit_loop: FitLoop):
        self.num_folds = num_folds
        self.fit_loop = fit_loop
        self.current_fold: int = 0
        self._callback_metrics = [ ]
    
    @property
    def done(self) -> bool:
        """ Check if we are done """
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        """Nothing to reset in this loop."""
        pass

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """
        Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store 
        the original weights of the model.
        """
        if not isinstance(self.trainer.datamodule, BaseKFoldDataModule):
            raise ValueError(
                'Expected the trainer to have an instance of the BaseKFoldDatamodule equipped'
                f' when running cross validation, but got {self.trainer.datamodule} instead'
            )
        # Setup the datasets for this fold
        self.trainer.datamodule.setup_folds()

        # Make a copy of the initial state of the model
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """ Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance. """
        rank_zero_info(f"Starting fold {self.current_fold+1}/{self.num_folds}")
        self.trainer.datamodule.setup_fold_index(self.current_fold)  

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.trainer.weights_save_path, f"model_fold{self.current_fold}.pt"))
        self._callback_metrics.append(deepcopy(self.trainer.callback_metrics))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.accelerator.setup_optimizers(self.trainer)
        
        """ Increment the logger version if possible """
        logger = self.trainer.logger
        if isinstance(logger, LoggerCollection):
            for l in logger: 
                if hasattr(l, "increment"): l.increment()
        elif isinstance(logger, LightningLoggerBase) and hasattr(logger, "increment"):
            self.trainer.logger.increment()

        self.current_fold += 1  # increment fold tracking number.

    def on_run_end(self):
        """ At the end of the run we summarize the results by saving the mean, standard diviation and
            raw values in the callback metrics attribute
        """
        self.trainer.logger_connector._callback_metrics = {}
        for k in self._callback_metrics[0].keys():
            values = torch.stack([cm[k] for cm in self._callback_metrics])
            self.trainer.logger_connector._callback_metrics[k+'_mean'] = values.mean()
            self.trainer.logger_connector._callback_metrics[k+'_std'] = values.std()
            self.trainer.logger_connector._callback_metrics[k+'_raw'] = values

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    @property
    def global_step(self) -> int:
        return self.fit_loop.global_step

    @global_step.setter
    def global_step(self, value) -> None:
        self.fit_loop.global_step = value

    @property
    def current_epoch(self) -> int:
        return self.fit_loop.current_epoch

    @current_epoch.setter
    def current_epoch(self, value) -> None:
        self.fit_loop.current_epoch = value

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]
