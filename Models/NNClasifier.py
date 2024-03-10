"""
Model specification for Binairy classifier model using static data

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09
"""

import lightning as L

# import torch
from torch import nn, rand
from torch.optim import Adam, AdamW
from torchmetrics.classification import BinaryAUROC

# %%


class BinaryClassifier(L.LightningModule):
    def __init__(self, input_size, layer_1_size, dropout_perc=0.2, lr=1e-3):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, int(layer_1_size / 2))
        self.fc3 = nn.Linear(int(layer_1_size / 2), 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_perc)
        self.save_hyperparameters()
        self.example_input_array = rand(7)
        self.lr = lr

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, temp, y = batch
        y_hat = self(x)
        loss = nn.functional.binary_cross_entropy(input=y_hat, target=y.float())
        AUC = BinaryAUROC()(y_hat, y.float())
        self.log_dict({"loss/train": loss, "auc/train": AUC})
        # self.log_dict({"loss/train": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        x, temp, y = batch
        y_hat = self(x)
        loss = nn.functional.binary_cross_entropy(input=y_hat, target=y.float())
        AUC = BinaryAUROC()(y_hat, y.float())
        self.log_dict({"loss/val": loss, "auc/val": AUC})
        # self.log_dict({"loss/val": loss})
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
