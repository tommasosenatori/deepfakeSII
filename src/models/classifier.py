from .resnet18 import build_resnet18
from .small_cnn import build_small_cnn
from .medium_cnn import build_medium_cnn

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,  
    MulticlassAccuracy,
    MulticlassF1Score
)


def build_backbone(model_name, num_outputs):

    if model_name == "resnet18":
        return build_resnet18(num_outputs=num_outputs)

    if model_name == "small_cnn":
        return build_small_cnn(num_outputs=num_outputs)

    if model_name == "medium_cnn":
        return build_medium_cnn(num_outputs=num_outputs)

    raise ValueError(f"Unknown model: {model_name}")


class ImageClassifier(pl.LightningModule):

    def __init__(
        self,
        model_name="resnet18",
        task="binary",
        num_classes=2,
        learning_rate=1e-4,
        weight_decay=1e-4
    ):
        super().__init__()

        self.save_hyperparameters()

        if task not in ["binary", "multiclass"]:
            raise ValueError("task must be either 'binary' or 'multiclass'.")

        self.model_name = model_name
        self.task = task
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if self.task == "binary":
            num_outputs = 1
        else:
            num_outputs = self.num_classes

        self.model = build_backbone(
            model_name=model_name,
            num_outputs=num_outputs
        )

        if self.task == "binary":
            self.train_acc = BinaryAccuracy()
            self.val_acc = BinaryAccuracy()
            self.test_acc = BinaryAccuracy()

            self.train_f1 = BinaryF1Score()
            self.val_f1 = BinaryF1Score()
            self.test_f1 = BinaryF1Score()

        else:
            self.train_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
            self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
            self.test_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")

            self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
            self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
            self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")

    def forward(self, x):
        logits = self.model(x)

        if self.task == "binary":
            logits = logits.squeeze(1)

        return logits

    def _shared_step(self, batch, stage):

        images, labels = batch
        logits = self(images)

        if self.task == "binary":
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels.float()
            )

            probs = torch.sigmoid(logits)

            if stage == "train":
                self.train_acc.update(probs, labels)
                self.train_f1.update(probs, labels)

                self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("train_f1", self.train_f1, prog_bar=False, on_step=False, on_epoch=True)

            elif stage == "val":
                self.val_acc.update(probs, labels)
                self.val_f1.update(probs, labels)

                self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_f1", self.val_f1, prog_bar=False, on_step=False, on_epoch=True)

            elif stage == "test":
                self.test_acc.update(probs, labels)
                self.test_f1.update(probs, labels)

                self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("test_f1", self.test_f1, prog_bar=False, on_step=False, on_epoch=True)

        else:
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if stage == "train":
                acc = self.train_acc(preds, labels)
                f1 = self.train_f1(preds, labels)

                self.log("train_loss", loss, prog_bar=True, on_epoch=True)
                self.log("train_acc", acc, prog_bar=True, on_epoch=True)
                self.log("train_macro_f1", f1, prog_bar=False, on_epoch=True)

            elif stage == "val":
                acc = self.val_acc(preds, labels)
                f1 = self.val_f1(preds, labels)

                self.log("val_loss", loss, prog_bar=True, on_epoch=True)
                self.log("val_acc", acc, prog_bar=True, on_epoch=True)
                self.log("val_macro_f1", f1, prog_bar=True, on_epoch=True)

            elif stage == "test":
                acc = self.test_acc(preds, labels)
                f1 = self.test_f1(preds, labels)

                self.log("test_loss", loss, prog_bar=True, on_epoch=True)
                self.log("test_acc", acc, prog_bar=True, on_epoch=True)
                self.log("test_macro_f1", f1, prog_bar=True, on_epoch=True)

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


class BinaryImageClassifier(ImageClassifier):

    def __init__(self, model_name="resnet18", learning_rate=1e-4, weight_decay=1e-4):
        super().__init__(
            model_name=model_name,
            task="binary",
            num_classes=2,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )