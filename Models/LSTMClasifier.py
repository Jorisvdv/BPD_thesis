"""
Model specification for LSTM Classifier model

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09
"""

# %%

import lightning as L
from Models.AE_CNN import CNNAutoencoder, CNNAutoencoderWithLoss
from torch import nn, rand, stack
from torch.optim import Adam
from torchmetrics.classification import BinaryAUROC

# %%


class LSTMClassifier(L.LightningModule):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        num_layers,
        Aencoder,
        dropout_perc=0.0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["Aencoder"])
        # self.encoder = nn.LSTM(
        #     input_size, hidden_size, num_layers, dropout=dropout_perc, batch_first=True
        # )
        self.AEncoder = Aencoder
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout_perc,
            batch_first=True,
        )
        self.fc2 = nn.Linear(
            hidden_size * num_layers, int(hidden_size * num_layers / 2)
        )
        self.fc3 = nn.Linear(int(hidden_size * num_layers / 2), 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.loss = nn.BCELoss()
        self.dropout = nn.Dropout(dropout_perc)
        # self.example_input_array = rand(1, 84, 120, input_size)
        self.lr = lr
        self.TrainAUC = BinaryAUROC()
        self.ValAUC = BinaryAUROC()

    def forward(self, sequences):
        # Loop through each patient individually because a LSTM can only recieve a 3D tensor
        encodings_list = list()
        for i in range(sequences.shape[0]):
            # Special case because CNN AE needs to permute dimensions
            if isinstance(self.AEncoder, CNNAutoencoder) or isinstance(
                self.AEncoder, CNNAutoencoderWithLoss
            ):
                encodings_list.append(self.AEncoder.encode(sequences[i]))
            else:
                encodings_list.append(self.AEncoder.encoder(sequences[i]))
        encodings_tensor = stack(encodings_list).squeeze()
        _, (hidden, _) = self.lstm(encodings_tensor)
        x = self.dropout(
            self.relu(self.fc2(hidden.permute(1, 0, 2).flatten(start_dim=1)))
        )
        # x = self.dropout(self.relu(self.fc2(hidden.squeeze(0))))
        x = self.sigmoid(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, seq, y = batch
        y_hat = self(seq)

        # y_hat = self(seq.squeeze(0))
        loss = nn.functional.binary_cross_entropy(input=y_hat, target=y.float())
        self.TrainAUC.update(preds=y_hat, target=y.float())
        self.log_dict({"loss/train": loss})
        # self.log_dict({"loss/train": loss})
        return loss

    def on_train_epoch_end(self):
        AUC = self.TrainAUC.compute()
        self.log_dict({"auc/train": AUC})
        self.TrainAUC.reset()

    def validation_step(self, batch, batch_idx):
        x, seq, y = batch
        y_hat = self(seq)
        loss = nn.functional.binary_cross_entropy(input=y_hat, target=y.float())
        self.ValAUC.update(preds=y_hat, target=y.float())
        self.log_dict({"loss/val": loss})
        # self.log_dict({"loss/val": loss})
        return loss

    def on_validation_epoch_end(self):
        AUC = self.ValAUC.compute()
        self.log_dict({"auc/val": AUC})
        self.ValAUC.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class LSTMClassifierWithLoss(LSTMClassifier):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        num_layers,
        Aencoder,
        dropout_perc=0.0,
        lr=1e-3,
    ):
        super().__init__(
            input_size=input_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            Aencoder=Aencoder,
            dropout_perc=dropout_perc,
            lr=lr,
        )
        # Override size to account for loss value
        self.lstm = nn.LSTM(
            embedding_size + 1,
            hidden_size,
            num_layers,
            dropout=dropout_perc,
            batch_first=True,
        )

    def forward(self, sequences):
        # Loop through each patient individually because a LSTM can only recieve a 3D tensor
        encodings_loss_list = list()
        for i in range(sequences.shape[0]):
            encodings_loss_list.append(self.AEncoder.encoding_and_loss(sequences[i]))
        encodings_tensor = stack(encodings_loss_list, dim=0)
        _, (hidden, _) = self.lstm(encodings_tensor)
        x = self.dropout(
            self.relu(self.fc2(hidden.permute(1, 0, 2).flatten(start_dim=1)))
        )
        x = self.sigmoid(self.fc3(x))
        return x
