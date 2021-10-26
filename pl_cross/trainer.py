from pytorch_lightning import Trainer as Trainer_pl


class Trainer(Trainer_pl):
    """ 
    Specialized trainer that implements additional methods for easy cross validation
    in pytorch lightning
    
    Args:
        K: number of folds for cross validation
        stratified: boolean indicating if folds should be constructed in a
            stratified way. Currently only supported if you dataset has a `labels`
            attribute.
        *args: additional arguments to pass to normal trainer constructor
        **kwargs: additional keyword arguments to pass to normal trainer constructor
    """
    def __init__(self, K: int = 5, stratified: bool = False, *args, **kwargs):
        self.K = K
        self.stratified = stratified
        super().__init__(*args, **kwargs)

