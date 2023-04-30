import torch
from lightning.pytorch import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset


class RandomDictDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self):
        return self.len


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class RandomLabelDataset(RandomDataset):
    def __init__(self, size: int, length: int):
        super().__init__(size, length)
        self.target = torch.randint(2, (length,))

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class RandomDictLabelDataset(RandomDictDataset):
    def __getitem__(self, index):
        a = self.data[index]
        b = torch.randint(2, (1,))
        return {"a": a, "b": b}


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self):
        return self.count


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class BoringDataModule(LightningDataModule):
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
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)

    def predict_dataloader(self):
        return DataLoader(self.random_predict)


class LitClassifier(LightningModule):
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

    def _step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", (y_hat.argmax(dim=-1) == y).float().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("valid_loss", loss)
        self.log("valid_acc", (y_hat.argmax(dim=-1) == y).float().mean())

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", (y_hat.argmax(dim=-1) == y).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
