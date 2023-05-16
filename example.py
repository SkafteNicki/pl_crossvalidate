import torch
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from pl_crossvalidate import KFoldTrainer


class LitClassifier(LightningModule):
    """Basic MNIST classifier."""

    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        """Forward pass."""
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def _step(self, batch):
        """Shared step function."""
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, y_hat, y = self._step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", (y_hat.argmax(dim=-1) == y).float().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, y_hat, y = self._step(batch)
        self.log("valid_loss", loss)
        self.log("valid_acc", (y_hat.argmax(dim=-1) == y).float().mean())

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, y_hat, y = self._step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", (y_hat.argmax(dim=-1) == y).float().mean())

    def score(self, batch, batch_idx):
        """Specialized score function that is used to calculate out of sample predictions."""
        x, y = batch
        return self(x).softmax(dim=-1)

    def configure_optimizers(self):
        """Configure optimizer for runs."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    # Setup data
    dataset_train = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    dataset_test = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    dataset_train, dataset_valid = random_split(dataset_train, [55000, 5000])
    train_dataloader = DataLoader(dataset_train, batch_size=64)
    val_dataloader = DataLoader(dataset_valid, batch_size=64)
    test_dataloader = DataLoader(dataset_test, batch_size=64)

    # Setup model
    model = LitClassifier()

    # Setup trainer
    trainer = KFoldTrainer(max_epochs=1, num_folds=2)

    # Cross validation
    output = trainer.cross_validate(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    print(output)

    # Out of sample scoring
    oos_score = trainer.out_of_sample_score(model)
    print(oos_score)
