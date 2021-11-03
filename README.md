# pl_cross

Cross validation in pytorch lightning made easy :]

Just import the specialized trainer from `pl_cross` instead of `pytorch_lightning` and you are set
```python
from pl_cross import Trainer

model = MyModel(...)

datamodule = MyDatamodule(...)

# Takes all original arguments + three new for controling the cross validation
trainer = Trainer(
  K=5,  # number of folds to do 
  shuffle=False,  # if samples should be shuffled before splitting
  stratified=False,  # if splitting should be done in a stratified manner
  gpus=...,
  logger=...,
  callbacks=...,
  ...
)

trainer.cross_validate(mode, datamodule=datamodule)
```

## Installation

Requires pytorch-lightning v1.5 or newer.
Simply install with
```bash
pip install pl_cross
```

## Cross-validation: why?

## Some limitations

* Cross validation is always done sequentially, even if the device you are training on in principal could
fit parallel training on multiple folds at the same time

* Logging can be a bit weird. We internally adjust the `logger._prefix` attribute of your chosen logger to 
  be `fold_{i}{logger.LOGGER_JOIN_CHAR}{logger._prefix}`. Therefore, a metric called `train_loss` for fold
  0 will be logged under the name: `fold_0-train_loss` by default. Depending on what logger you are using
  it may be beneficial to change the `logger.LOGGER_JOIN_CHAR` attribute to log the same metric from different
  folds together (for example using tensorboard, it is recommended changing this to a slash e.g. `"/"`)

* Stratified splitting assume that we can extract a 1D label vector from your dataset.
  * If your dataset has an `labels` attribute, we will use that as the labels
  * If the attribute does not exist, we manually iterate over your dataset trying to
    extract the labels. By default we assume that given a `batch` the labels can be found
    as the second argument e.g. `batch[1]`. You can adjust this by initializing a 
    ```python
    from pl_cross import Trainer, KFoldDataModule
    
    model = ...

    trainer = Trainer(...)

    datamodule = KFoldDataModule(
      num_folds, shuffle, stratified,  # these should match how the trainer is initialized
      train_dataloader=my_train_dataloader,
    )
    # change the label extractor function, such that it will return the labels for a given batch
    datamodule.label_extractor = lambda batch: batch['y']

    trainer.cross_validate(model, datamodule=datamodule)
    ```
## 

