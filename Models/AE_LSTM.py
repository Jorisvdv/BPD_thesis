"""
Model specification for LSTM Autoencoder

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09
"""

# %%

import lightning as L

# import torch
from torch import abs, cat, nn, rand, square
from torch.optim import Adam, AdamW


# %%
class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        # x.detach()
        x, (hidden_n, _) = self.rnn2(x)
        # hidden_n.detach()
        return hidden_n.reshape((x.shape[0], -1))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        # x.detach()
        x, (hidden_n, cell_n) = self.rnn2(x)
        # x.detach()
        return self.output_layer(x)


class RecurrentAutoencoder(L.LightningModule):
    def __init__(self, seq_len, n_features, embedding_dim=64, lr=1e-3, loss="L1"):
        super(RecurrentAutoencoder, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)
        self.lr = lr
        self.example_input_array = rand(1000, seq_len, n_features)
        self.loss = nn.MSELoss() if loss == "MSE" else nn.L1Loss()
        self.latent_size = embedding_dim

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat = self(x)
        # loss = nn.functional.l1_loss(x_hat, x)  # nn.functional.mse_loss
        loss = self.loss(x_hat, x)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        x_hat = self(x)
        # loss = nn.functional.l1_loss(x_hat, x)  # nn.functional.mse_loss
        loss = self.loss(x_hat, x)
        self.log("loss/val", loss)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    # def backward(self, loss):
    #     loss.backward(retain_graph=True)


class RecurrentAutoencoderWithLoss(RecurrentAutoencoder):
    def encoding_and_loss(self, x):
        encoding = self.encoder(x)
        x_hat = self.decoder(encoding)
        loss = abs(x_hat - x).mean(dim=1)
        return cat([encoding, loss], dim=1)
        # return loss
