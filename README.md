# pl_cross

Cross validation in pytorch lightning made easy :]

Just import the specialized trainer from `pl_cross` instead of `pytorch_lightning` and you are set
```python
from pl_cross import Trainer

# Takes all original arguments + two new for controling the cross validation
trainer = Trainer(
  K=5, 
  stratified=False, 
  gpus=...,
  logger=...,
  callbacks=...,
  ...
)
```
