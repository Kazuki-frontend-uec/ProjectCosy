import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import whisper
from whisper.audio import pad_or_trim


# -----------------------------
#  Whisper Encoder 抽出ラッパ
# -----------------------------
class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        # load_model に device を渡しておくと model が自動でデバイスに乗る
        self.model = whisper.load_model(model_name, device=device)
        self.encoder = self.model.encoder
        self.mel_fn = whisper.log_mel_spectrogram

    @torch.no_grad()
    def forward(self, wav_batch):
        # wav_batch: [B, T]

        embs = []
        frame_lengths = []
        for wav in wav_batch:
            # 元のサンプル数
            orig_len = wav.shape[-1]

            # 30秒に padding か trimming
            wav_np = wav.detach().cpu().numpy()
            wav_np = pad_or_trim(wav_np)
            wav_trim = torch.from_numpy(wav_np).float().to(self.device)

            # melスペクトログラムに変換
            mel = self.mel_fn(wav_trim).to(self.device)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            # Encoderに通す
            emb = self.encoder(mel)  # [1, T, dim]
            embs.append(emb)

            # フレーム長計算、Whisperは hop_length=160
            frame_len = min(orig_len // 160, emb.shape[1])
            frame_lengths.append(frame_len)

            embs = torch.cat(embs, dim=0)  # [B, T, dim]
            frame_lengths = torch.tensor(frame_lengths, device=self.device)

        return embs, frame_lengths

