import torch
from lightning.pytorch import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset, Subset


class RandomDictDataset(Dataset):
    """A dataset that returns a dictionary of tensors."""

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        """Return sample."""
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self):
        """Return size of dataset."""
        return self.len


class RandomDataset(Dataset):
    """A dataset that returns a tensor."""

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        """Return sample."""
        return self.data[index]

    def __len__(self):
        """Return size of dataset."""
        return self.len


class RandomLabelDataset(RandomDataset):
    """A dataset that returns a tensor with labels."""

    def __init__(self, size: int, length: int):
        super().__init__(size, length)
        self.target = torch.randint(2, (length,))

    def __getitem__(self, index):
        """Return sample."""
        return self.data[index], self.target[index]


class RandomDictLabelDataset(RandomDictDataset):
    """A dataset that returns a dictionary of tensors with labels."""

    def __getitem__(self, index):
        """Return sample."""
        a = self.data[index]
        b = torch.randint(2, (1,))
        return {"a": a, "b": b}


class BoringModel(LightningModule):
    """A simple model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        """Forward pass."""
        return self.layer(x)

    def loss(self, batch, prediction):
        """A simple loss function."""
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        """Training step."""
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        """Test step."""
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def configure_optimizers(self):
        """Return whatever optimizers we want here."""
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        """Predict dataloader."""
        return DataLoader(RandomDataset(32, 64))


class BoringDataModule(LightningDataModule):
    """A simple datamodule with random data for testing purposes."""

    def __init__(self, with_labels=False, feature_size=32):
        self.with_labels = with_labels
        self.feature_size = feature_size

        dataclass = RandomLabelDataset if self.with_labels else RandomDataset
        self.random_full = dataclass(self.feature_size, 64 * 4)

        self.random_train = Subset(self.random_full, indices=range(64))
        self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))
        self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))
        self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(self.random_train)

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(self.random_val)

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(self.random_test)

    def predict_dataloader(self):
        """Predict dataloader."""
        return DataLoader(self.random_predict)


class LitClassifier(LightningModule):
    """Simple MNIST classifier."""

    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        """Forward pass for the model."""
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

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
