import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from whisper.audio import pad_or_trim
# from funasr import AutoModel

class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model = whisper.load_model(model_name, device=device)
        self.model.encoder.eval()
        self.encoder = self.model.encoder

    @torch.no_grad()
    def forward(self, wav):
        # wav: [Batch, T] or [T]
        # whisper.log_mel_spectrogram は Tensor を受け取れるので CPU/GPU 往復は不要
        mel = whisper.log_mel_spectrogram(wav).to(self.device)

        # 30秒(3000フレーム)に調整
        mel = whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        emb = self.encoder(mel)
        return emb # [1, 1500, 512] batchを維持して返すのが一般的

