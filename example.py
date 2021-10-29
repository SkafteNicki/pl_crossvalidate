# adjusted version of
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/simple_image_classifier.py
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from pl_cross import Trainer


class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    # Setup data
    dataset_train = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    dataset_test = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    dataset_train, dataset_valid = random_split(dataset_train, [55000, 5000])
    train_dataloader = DataLoader(dataset_train, batch_size=64)
    valid_dataloader = DataLoader(dataset_valid, batch_size=64)
    test_dataloader = DataLoader(dataset_test, batch_size=64)

    # Setup model
    model = LitClassifier()

    # Setup trainer
    trainer = Trainer(max_epochs=1, default_root_dir='models/')

    # Normal fitting
    #trainer.fit(model, train_dataloader, val_dataloaders=valid_dataloader)

    # Do cross validation
    trainer.cross_validate(model, train_dataloader, valid_dataloader)
