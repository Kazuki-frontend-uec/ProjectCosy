import torch
import torch.nn as nn
import torch.nn.functional as F

class AE_512to8(nn.Module):
    """
    単純なボトルネックオートエンコーダー
    Whisperのエンコード出力（512次元）を、8次元の潜在表現に圧縮して復元するモデル
    """
    def __init__(self, input_dim=512, latent_dim=8):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon
