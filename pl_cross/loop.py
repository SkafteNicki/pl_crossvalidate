import os.path as osp
from copy import deepcopy
from typing import Any, Dict

import logging
import torch
from pytorch_lightning.loggers.base import LoggerCollection
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
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        """Nothing to reset in this loop."""

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
        
        rank_zero_info(
            f"Starting fold {self.current_fold+1}/{self.num_folds} \n"
        )
        self.trainer.datamodule.setup_fold_index(self.current_fold)

        # hijack the _prefix argument of the users logger to correctly log metrics for each fold
        logger = self.trainer.logger
        logger = logger if isinstance(self.trainer.logger, LoggerCollection) else [logger]
        for l in logger:
            if not hasattr(l, '_orig_prefix'):
                l._orig_prefix = l._prefix
            prefix = f"{l.LOGGER_JOIN_CHAR}{l._orig_prefix}" if l._orig_prefix != '' else ''
            l._prefix = f"fold_{self.current_fold}{prefix}"

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.trainer.weights_save_path, f"model_fold{self.current_fold}.pt"))
        self._callback_metrics.append(deepcopy(self.trainer.callback_metrics))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.accelerator.setup_optimizers(self.trainer)

    def on_run_end(self):
        # Calculate average 
        self.trainer.logger_connector._callback_metrics = {}
        for k in self._callback_metrics[0].keys():
            values = torch.stack([cm[k] for cm in self._callback_metrics])
            self.trainer.logger_connector._callback_metrics[k+'_mean'] = values.mean()
            self.trainer.logger_connector._callback_metrics[k+'_std'] = values.std()
            self.trainer.logger_connector._callback_metrics[k+'_raw'] = values


#    def on_run_end(self) -> None:
#        """Used to compute the performance of the ensemble model on the test set."""
#        checkpoint_paths = [osp.join(self.trainer.weights_save_path, f"model_fold{f_idx}.pt") for f_idx in range(self.num_folds)]
#        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
#        voting_model.trainer = self.trainer
#        # This requires to connect the new model and move it the right device.
#        self.trainer.accelerator.connect(voting_model)
#        self.trainer.training_type_plugin.model_to_device()
#        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.fit_loop.current_epoch = 0
        self.fit_loop.global_step = 0
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]
