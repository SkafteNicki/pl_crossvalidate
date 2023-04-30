import os.path as osp
from typing import Any, List, Optional, Sequence, Union

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

from pl_cross.datamodule import KFoldDataModule
from pl_cross.ensemble import EnsembleLightningModule


class KFoldTrainer(Trainer):
    """Trainer for KFold cross validation.

    This trainer is a plug-in replacement for the `Trainer` class from `lightning` package. It does not alter any of
    the existing functionality, only extends it with the ability to perform KFold cross validation. In total three
    new public methods are added:

    - `cross_validate`: Performs KFold cross validation on a given model
    - `create_ensemble`: Creates an ensemble model from a list of trained models
    - `out_of_sample_score`: Computes the out-of-sample score for a given model

    The two latter of the methods are intended to be called after the `cross_validate` method has been called.

    Args:
        num_folds: Number of folds
        shuffle: Whether to shuffle the data before splitting it into folds
        stratified: Whether to use stratified sampling e.g. for classification we make sure that each fold has the same
            ratio of samples from each class as the original dataset
        args: Arguments passed to the underlying lightning `Trainer` class
        kwargs: Keyword arguments passed to the underlying lightning  `Trainer` class
    """

    def __init__(self, num_folds: int = 5, shuffle: bool = False, stratified: bool = False, *args, **kwargs) -> None:
        # Input validation for the cross validation arguments
        if not isinstance(num_folds, int) or num_folds < 2:
            raise ValueError("Expected argument `num_folds` to be an integer larger than or equal to 2")
        self.num_folds = num_folds
        if not isinstance(shuffle, bool):
            raise ValueError("Expected argument `shuffle` to be an boolean")
        self.shuffle = shuffle
        if not isinstance(stratified, bool):
            raise ValueError("Expected argument `stratified` to be an boolean")
        self.stratified = stratified

        # Intialize rest of the trainer
        super().__init__(*args, **kwargs)
        self._version = self.logger.version

    def _construct_kfold_datamodule(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, Sequence[DataLoader]]] = None,
        datamodule: Optional[Union[LightningDataModule, KFoldDataModule]] = None,
    ) -> KFoldDataModule:
        return KFoldDataModule(
            self.num_folds,
            self.shuffle,
            self.stratified,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
        )

    def cross_validate(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, Sequence[DataLoader]]] = None,
        datamodule: Optional[Union[LightningDataModule, KFoldDataModule]] = None,
    ) -> List[Any]:
        """Performs KFold cross validation on a given model.

        This is the core method added by this class. Given a model and a dataloader / datamodule it will automatically
        perform KFold cross validation. By automatically we mean that it will automatically construct the different
        folds and sequentially train and validate the model on each fold, resetting the model weights between each fold.

        Args:
            model: The model to perform cross validation on
            train_dataloader: Dataloader with training samples
            val_dataloaders: Dataloader with validation samples, can be a list of dataloaders for multiple validation
            datamodule: Lightning datamodule (exclusive with `train_dataloader` and `val_dataloader`)
        """
        if not is_overridden("test_step", model, LightningModule):
            raise ValueError("`cross_validation` method requires you to also define a `test_step` method.")

        # construct K fold datamodule if user is not already passing one in
        if not isinstance(datamodule, KFoldDataModule):
            datamodule = self._construct_kfold_datamodule(train_dataloader, val_dataloaders, datamodule)
        self._kfold_datamodule = datamodule

        # checkpoint to restore from
        # this is a bit hacky because the model needs to be saved before the fit method
        self.strategy._lightning_module = model
        path = osp.join(self.log_dir, "kfold_initial_weights.ckpt")
        self.save_checkpoint(path)
        self.strategy._lightning_module = None

        # run cross validation
        results, paths = [], []
        for i in range(self.num_folds):
            self._set_fold_index(i, datamodule=datamodule)
            print(self.logger.log_dir)
            rank_zero_info(f"===== Starting fold {i+1}/{self.num_folds} =====")
            self.fit(model=model, datamodule=datamodule, ckpt_path=path)

            path = osp.join(self.log_dir, f"fold_{i}.ckpt")
            self.save_checkpoint(path)
            paths.append(path)

            res = self.test(model=model, datamodule=datamodule, verbose=False)
            results.append(res)

        self._ensemple_paths = paths
        return results

    def _set_fold_index(self, fold_index: int, datamodule: KFoldDataModule) -> None:
        # any logger need to be reset to the new fold index and the privious experiment needs to be cleared
        if self.loggers is not None:
            for logger in self.loggers:
                if hasattr(logger, "_version"):
                    new_version = f"{self._version}/fold_{fold_index}" if self._version else f"fold_{fold_index}"
                    logger._version = new_version
                if hasattr(logger, "_experiment"):
                    logger._experiment = None

        # set the fold index for the datamodule
        datamodule.fold_index = fold_index

    def create_ensemble(self, model: LightningModule, ckpt_paths: Optional[List[str]] = None) -> LightningModule:
        """Create an ensemble from trained models.

        Args:
            model: An instance of the model to create an ensemble over.
            ckpt_paths: If not provided, then it assumes that `cross_validate` have been already called
                and will automatically load the model checkpoints saved during that process. Else expect
                it to be a list of checkpoint paths to individual models.

        Example:
            >>> trainer = Trainer()
            >>> trainer.cross_validate(model, datamodule)
            >>> ensemble_model = trainer.create_ensemble(model)

        """
        if ckpt_paths is None:
            if hasattr(self, "_ensemple_paths"):
                ckpt_paths = self._ensemple_paths
            else:
                raise ValueError(
                    "Cannot construct ensemble model. Either call `cross_validate`"
                    "beforehand or pass in a list of ckeckpoint paths in the `ckpt_paths` argument"
                )
        return EnsembleLightningModule(model, ckpt_paths)

    def out_of_sample_score(
        self,
        model: LightningModule,
        datamodule: Optional[KFoldDataModule] = None,
        ckpt_paths: Optional[List[str]] = None,
    ) -> LightningModule:
        """Performs out of sample scoring for a given set of KFold trained models.

        Out of sample scoring for K-fold refer to predicting on the test fold of each K-fold model.

        Args:
            model: The model to perform cross validation on
            datamodule: Optionally a ``KFoldDataModule`` instance. If not provided, then it assumes that
                `cross_validate` have been already called and will automatically use the same ``KFoldDataModule`` use
                during that process.
            ckpt_paths: If not provided, then it assumes that `cross_validate` have been already called
                and will automatically load the model checkpoints saved during that process. Else expect
                it to be a list of checkpoint paths to individual models.
        """
        score_method = getattr(model, "score", None)
        if not callable(score_method):
            raise ValueError("`out_of_sample_score` method requires you to also define a `score` method.")

        if ckpt_paths is None:
            if hasattr(self, "_ensemple_paths"):
                ckpt_paths = self._ensemple_paths
            else:
                raise ValueError(
                    "Cannot construct ensemble model. Either call `cross_validate`"
                    "beforehand or pass in a list of ckeckpoint paths in the `ckpt_paths` argument"
                )

        if datamodule is None:
            if not hasattr(self, "_kfold_datamodule"):
                raise ValueError(
                    "Cannot compute out of sample scores. Either call `cross_validate` method before"
                    "`out_of_sample_score` method, or provide an instance of `KFoldDataModule` in the `datamodule`"
                    "argument."
                )
            else:
                datamodule = self._kfold_datamodule
        elif not isinstance(datamodule, KFoldDataModule):
            raise ValueError("`datamodule` argument must be an instance of `KFoldDataModule`.")

        if len(ckpt_paths) != datamodule.num_folds:
            raise ValueError("Number of checkpoint paths provided does not match the number of folds in the datamodule")

        # temporarily replace the predict_step method with the score method to use the trainer.predict method
        _orig_predict_method = model.predict_step
        model.predict_step = model.score

        # run prection on each fold
        outputs = []
        for i, ckpt_path in enumerate(ckpt_paths):
            self._set_fold_index(i, datamodule=datamodule)
            model.load_from_checkpoint(ckpt_path)
            out = self.predict(model=model, dataloaders=datamodule.test_dataloader())
            outputs.append(torch.cat(out, 0))
        model.predict_step = _orig_predict_method

        # reorder to match the order of the dataset
        test_indices = torch.cat([torch.tensor(test) for _, test in datamodule.splits])
        outputs = torch.cat(outputs, 0)
        return outputs[test_indices.argsort()]
