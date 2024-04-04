import os
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics import Metric, MetricCollection, Accuracy, Precision, Recall, AUROC, AveragePrecision, MeanMetric, MaxMetric
import pickle
from torch.utils.data import DataLoader, TensorDataset

class Model(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimization_params: Dict | DictConfig,
        n_classes: int
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(optimization_params["lr"])
        self.step_size = int(optimization_params["lr_scheduler_step_size"])
        self.gamma = float(optimization_params["lr_scheduler_gamma"])
        self.weight_decay = float(optimization_params["l2_coeff"])
        self.optimizer, self.scheduler = self.configure_optimizers()
        self.logger: MLFlowLogger
        self.trainer: pl.Trainer

        if n_classes > 2:
            metrics_params = {"task": "multiclass", "num_classes": n_classes}

            metrics = MetricCollection([
                Accuracy(**metrics_params),
                Recall(**metrics_params),
                Precision(**metrics_params),
                AUROC(**metrics_params),
                AveragePrecision(**metrics_params),
            ])

            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
            self.val_acc_best = MaxMetric()
        else:
            raise ValueError("Number of classes must be greater than 2.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Mapping[str, torch.Tensor]:
        logits = self.forward(batch[0])
        loss = F.cross_entropy(logits, batch[1].long())
        self.train_loss(loss)
        self.train_metrics(logits, batch[1])
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        logits = self.forward(batch[0])
        loss = F.cross_entropy(logits, batch[1].long())
        self.val_loss(loss)
        self.val_metrics(logits, batch[1])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, torch.optim.lr_scheduler._LRScheduler]]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)}
        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self.log_dict(self.train_metrics.compute(), on_epoch=True, prog_bar=True)
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        self.val_loss.reset()
        self.val_metrics.reset()
        if not self.trainer.sanity_checking:
            if len(outputs) > 0:
                if len(outputs[0]["targets"].shape) == 1:
                    acc = self.val_metrics.compute()["val_Accuracy"]
                else:
                    acc = self.val_metrics.compute()["val_MulticlassAccuracy"]
                self.val_acc_best(acc)
                self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)



class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_file, batch_size=64):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        with open(self.data_file, 'rb') as f:
            xtrain, xvalid, ytrain, yvalid = pickle.load(f)

        split_index = len(xvalid) // 2
        xtest, ytest = xvalid[:split_index], yvalid[:split_index]
        xvalid, yvalid = xvalid[split_index:], yvalid[split_index:]

        self.train_dataset = TensorDataset(torch.tensor(xtrain), torch.tensor(ytrain))
        self.val_dataset = TensorDataset(torch.tensor(xvalid), torch.tensor(yvalid))
        self.test_dataset = TensorDataset(torch.tensor(xtest), torch.tensor(ytest))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
