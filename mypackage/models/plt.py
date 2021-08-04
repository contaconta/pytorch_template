import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy

class LitModel(pl.LightningModule):
    def __init__(self, base_model, lr: float = 1e-2):
        super().__init__()
        self.save_hyperparameters('lr')
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc1 = accuracy(y_hat, y, top_k=1)
        acc5 = accuracy(y_hat, y, top_k=5)
        metrics = {"train_acc": acc1, "train_acc_top5": acc5, "train_loss": loss}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc1 = accuracy(y_hat, y, top_k=1)
        acc5 = accuracy(y_hat, y, top_k=5)
        metrics = {"val_acc": acc1, "val_acc_top5": acc5, "train_loss": loss}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
