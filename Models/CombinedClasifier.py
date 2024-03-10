"""
Model specification for Classifier model using a combination of static and temporal data

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09
"""

# %%

import lightning as L
from torch import cat, nn, rand, stack
from torch.optim import Adam
from torchmetrics.classification import BinaryAUROC

# %%


class CombinedClassifier(L.LightningModule):
    def __init__(
        self,
        num_temoral_features,  # Added for future definition of encoder
        num_static_featues,
        static_layer_size,
        embedding_size,
        hidden_size,
        num_layers,
        AEncoder,
        dropout_perc=0.0,
        lr=1e-3,
    ):
        super().__init__()
        # self.save_hyperparameters(ignore=["AEncoder"])
        # self.encoder = nn.LSTM(
        #     input_size, hidden_size, num_layers, dropout=dropout_perc, batch_first=True
        # )
        self.Encoder = AEncoder.encoder

        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout_perc,
            batch_first=True,
        )
        self.static_fc1 = nn.Linear(num_static_featues, static_layer_size)
        self.static_fc2 = nn.Linear(static_layer_size, int(static_layer_size / 2))

        self.temporal_fc2 = nn.Linear(
            hidden_size * num_layers, int(hidden_size * num_layers / 2)
        )

        self.fc_combined = nn.Linear(
            int((hidden_size * num_layers / 2) + (static_layer_size / 2)), 1
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.loss = nn.BCELoss()
        self.dropout = nn.Dropout(dropout_perc)
        # self.example_input_array = rand(1, 84, 120, num_temoral_features)
        self.lr = lr

    def forward(self, static, sequences):
        # Proces temporal data
        # Loop through each patient individually because a LSTM can only recieve a 3D tensor
        encodings_list = list()
        for i in range(sequences.shape[0]):
            encodings_list.append(self.Encoder(sequences[i]))
        encodings_tensor = stack(encodings_list)
        _, (hidden, _) = self.lstm(encodings_tensor)
        x_em = self.dropout(self.relu(self.temporal_fc2(hidden.squeeze(0))))

        # Add static data
        x_fc = self.dropout(self.relu(self.static_fc1(static)))
        x_fc = self.dropout(self.relu(self.static_fc2(x_fc)))

        # Merge both models
        x_combined = cat((x_em, x_fc), dim=1)
        x_hat = self.sigmoid(self.fc_combined(x_combined))

        return x_hat

    def training_step(self, batch, batch_idx):
        x, seq, y = batch
        y_hat = self(x, seq)
        loss = nn.functional.binary_cross_entropy(y_hat, y.float())
        AUC = BinaryAUROC()(y_hat, y.float())

        self.log_dict({"loss/train": loss, "auc/train": AUC})
        # self.log_dict({"loss/train": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, seq, y = batch
        y_hat = self(x, seq)
        loss = nn.functional.binary_cross_entropy(y_hat, y.float())
        AUC = BinaryAUROC()(y_hat, y.float())

        self.log_dict({"loss/val": loss, "auc/val": AUC})
        # self.log_dict({"loss/val": loss})
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class CombinedClassifierWithLoss(CombinedClassifier):
    def __init__(
        self,
        num_temoral_features,  # Added for future definition of encoder
        num_static_featues,
        static_layer_size,
        embedding_size,
        hidden_size,
        num_layers,
        AEncoder,
        dropout_perc=0.0,
        lr=1e-3,
    ):
        super().__init__(
            num_temoral_features,
            num_static_featues,
            static_layer_size,
            embedding_size,
            hidden_size,
            num_layers,
            AEncoder,
            dropout_perc,
            lr,
        )
        self.AEncoder = AEncoder
        self.lstm = nn.LSTM(
            embedding_size + 1,
            hidden_size,
            num_layers,
            dropout=dropout_perc,
            batch_first=True,
        )
        del self.Encoder
        self.save_hyperparameters(ignore=["AEncoder"])

    def forward(self, static, sequences):
        # Proces temporal data
        # Loop through each patient individually because a LSTM can only recieve a 3D tensor
        encodings_list = list()
        for i in range(sequences.shape[0]):
            encodings_list.append(self.AEncoder.encoding_and_loss(sequences[i]))
        encodings_tensor = stack(encodings_list)
        _, (hidden, _) = self.lstm(encodings_tensor)
        x_em = self.dropout(self.relu(self.temporal_fc2(hidden.squeeze(0))))

        # Add static data
        x_fc = self.dropout(self.relu(self.static_fc1(static)))
        x_fc = self.dropout(self.relu(self.static_fc2(x_fc)))

        # Merge both models
        x_combined = cat((x_em, x_fc), dim=1)
        x_hat = self.sigmoid(self.fc_combined(x_combined))

        return x_hat


class CombinedClassifierCNN(CombinedClassifier):
    def __init__(
        self,
        num_temoral_features,  # Added for future definition of encoder
        num_static_featues,
        static_layer_size,
        embedding_size,
        hidden_size,
        num_layers,
        AEncoder,
        dropout_perc=0.0,
        lr=1e-3,
    ):
        super().__init__(
            num_temoral_features,
            num_static_featues,
            static_layer_size,
            embedding_size,
            hidden_size,
            num_layers,
            AEncoder,
            dropout_perc,
            lr,
        )
        self.AEncoder = AEncoder
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout_perc,
            batch_first=True,
        )
        del self.Encoder
        del self.fc_combined
        self.fc_combined_1 = nn.Linear(
            int((hidden_size * num_layers / 2) + (static_layer_size / 2)), 64
        )
        self.fc_combined_2 = nn.Linear(64, 32)
        self.fc_combined_3 = nn.Linear(32, 1)
        self.save_hyperparameters(ignore=["AEncoder"])

    def forward(self, static, sequences):
        # Proces temporal data
        # Loop through each patient individually because a LSTM can only recieve a 3D tensor
        encodings_list = list()
        for i in range(sequences.shape[0]):
            encodings_list.append(self.AEncoder.encode(sequences[i]))
        encodings_tensor = stack(encodings_list)
        _, (hidden, _) = self.lstm(encodings_tensor.squeeze())
        x_em = self.dropout(
            self.relu(self.temporal_fc2(hidden.permute(1, 0, 2).flatten(start_dim=1)))
        )

        # Add static data
        x_fc = self.dropout(self.relu(self.static_fc1(static)))
        x_fc = self.dropout(self.relu(self.static_fc2(x_fc)))

        # Merge both models
        x_combined = cat((x_em, x_fc), dim=1)
        x_combined = self.dropout(self.fc_combined_1(x_combined))
        x_combined = self.dropout(self.fc_combined_2(x_combined))
        y_hat = self.sigmoid(self.fc_combined_3(x_combined))

        return y_hat
