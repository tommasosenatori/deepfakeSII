from .resnet18 import build_resnet18
from .small_cnn import build_small_cnn
from .medium_cnn import build_medium_cnn

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score, BinaryAUROC)

def build_backbone(model_name):

    if model_name == "resnet18":
        return build_resnet18()

    if model_name == "small_cnn":
        return build_small_cnn()

    if model_name == "medium_cnn":
        return build_medium_cnn()

    raise ValueError(
        f"Unknown model: {model_name}"
    )


class BinaryImageClassifier(pl.LightningModule):

    def __init__(self, model_name="resnet18", learning_rate=1e-4, weight_decay=1e-4):
        super().__init__()

        self.save_hyperparameters()

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = build_backbone(model_name)

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

    def forward(self, x):
        logits = self.model(x)
        logits = logits.squeeze(1)
        return logits

    def _shared_step(self, batch, stage):
        images, labels = batch
        labels_float = labels.float()

        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels_float)

        probs = torch.sigmoid(logits)

        if stage == "train":
            acc = self.train_acc(probs, labels)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            self.log("train_acc", acc, prog_bar=True, on_epoch=True)

        elif stage == "val":
            acc = self.val_acc(probs, labels)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        elif stage == "test":
            acc = self.test_acc(probs, labels)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        return optimizer