import torch
import torch.nn as nn
import whisper
from vector_quantize_pytorch import FSQ

class WhisperFSQEncoder(nn.Module):
    def __init__(self, levels=[3]*8, device="cuda", model_name="base"):
        super().__init__()
        self.device = torch.device(device)

        # 1. 既存のEncoder、Whisperのロード
        self.whisper_model = whisper.load_model(model_name, device=device)
        self.encoder = self.whisper_model.encoder
        hidden_dim = self.encoder.ln_post.weight.shape[0] # 512

        # 2. FSQモジュール
        self.fsq = FSQ(levels=levels) # 3^8 = 6561
        dim_fsq = len(levels) # 8次元

        # 3. Projector層の学習
        self.project_in = nn.Linear(hidden_dim, dim_fsq)
        self.project_out = nn.Linear(dim_fsq, hidden_dim)

    def forward(self, wav):
        # ASR用のメルスペクトログラム取得
        mel = whisper.log_mel_spectrogram(wav).to(self.device)
        mel = whisper.audio.pad_or_trim(mel, 3000)

        if mel.dim() == 2:
            # [80, 3000] -> [1, 80, 3000]
            mel = mel.unsqueeze(0)
        elif mel.dim() == 4:
            # [Batch, 1, 80, 3000] -> [Batch, 80, 3000] になっている場合
            mel = mel.squeeze(1)

        # Encoder出力 [Batch, Seq, 512]
        with torch.no_grad():
            features = self.encoder(mel)

        # FSQの低次元空間へ投影 -> 量子化 -> 元の次元へ戻す
        z = self.project_in(features)
        z_q, indices = self.fsq(z)      # STE、勾配が維持される
        recon_features = self.project_out(z_q)

        return features, recon_features, indices
