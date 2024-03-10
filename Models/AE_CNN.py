"""
Model specification for CNN Autoencoder

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09
"""

import lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam


class CNNAutoencoder(L.LightningModule):
    def __init__(
        self,
        n_features,
        seq_len,
        latent_size,
        lr=1e-3,
        loss="L1",
    ):
        super(CNNAutoencoder, self).__init__()
        self.save_hyperparameters()
        self.input_channels = n_features
        self.input_length = seq_len
        self.latent_size = latent_size
        self.lr = lr
        self.example_input_array = torch.rand(1000, seq_len, n_features)
        self.loss = nn.MSELoss() if loss == "MSE" else nn.L1Loss()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(
                n_features, 16, kernel_size=5, stride=2, padding=2
            ),  # Reduce by half
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # Reduce by half
            nn.ReLU(),
            nn.Conv1d(
                32, latent_size, kernel_size=seq_len // 4
            ),  # Adjust the kernel size to fit the desired latent size
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_size, 32, kernel_size=seq_len // 4),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 16, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, n_features, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
        )

    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
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


class CNNAutoencoderPooled(L.LightningModule):
    def __init__(
        self,
        n_features,
        seq_len,
        latent_size,
        lr=1e-3,
        loss="L1",
        dropout=0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_channels = n_features
        self.input_length = seq_len
        self.latent_size = latent_size
        self.lr = lr
        self.example_input_array = torch.rand(1000, seq_len, n_features)
        self.loss = nn.MSELoss() if loss == "MSE" else nn.L1Loss()
        self.dropout = dropout

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_channels, self.latent_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Reduce by half
            nn.Dropout(self.dropout),
            nn.Conv1d(self.latent_size, self.latent_size * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Reduce by half
            nn.Dropout(self.dropout),
            nn.Conv1d(
                self.latent_size * 2,
                self.latent_size,
                kernel_size=self.input_length // 4,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                self.latent_size,
                self.latent_size * 2,
                kernel_size=self.input_length // 4,
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Increase by double
            nn.Dropout(self.dropout),
            nn.ConvTranspose1d(
                self.latent_size * 2, self.latent_size, kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Increase by double
            nn.Dropout(self.dropout),
            nn.ConvTranspose1d(
                self.latent_size, self.input_channels, kernel_size=5, padding=2
            ),
            nn.ReLU(),
        )

    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
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


class CNNAutoencoderWithLoss(CNNAutoencoder):
    def encoding_and_loss(self, x):
        encoding = self.encode(x)
        x_hat = self.decode(encoding)
        loss = torch.abs(x_hat - x).mean(dim=1).sum(dim=1).unsqueeze(dim=-1)
        return torch.cat([encoding.squeeze(), loss], dim=1)
