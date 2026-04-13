import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torchaudio
from WhisperEncoder import WhisperEncoder

class FSQWhisperTokenizer(nn.Module):
    """
    純粋に FSQ (no learning, no codebook, no EMA) のみでの実装版
    入力:
        x: [B, T, dim]
    出力:
        recon: [B, T, dim] 量子化後
        indices: [B, T, D] 各factorのindex
    """
    def __init__(self, num_factors=8, K=3, dim=512, vmin=-5.0, vmax=5.0):
        super().__init__()
        assert dim % num_factors == 0

        self.D = num_factors
        self.K = K
        self.dim = dim
        self.subdim = dim // num_factors

        # 固定レンジ
        self.vmin = vmin
        self.vmax = vmax
        # ステップ幅
        self.delta = (vmax - vmin) / (K - 1)


    def forward(self, x):
        # x: [B, T, dim]
        B, T, _ = x.shape

        # [B, T, D, subdim]
        x = x.view(B, T, self.D, self.subdim)

        # 正規化
        x_clipped = torch.clamp(x, self.vmin, self.vmax)

        # [0, K-1] にスケーリング
        x_scaled = (x_clipped - self.vmin) / self.delta

        # 離散化（round）
        idx = torch.round(x_scaled).long()
        idx = torch.clamp(idx, 0, self.K - 1)

        # 復元
        x_quant = idx.float() * self.delta + self.vmin
        # reshape戻す
        recon = x_quant.view(B, T, self.dim)

        # indices: [B, T, D]
        # 各factorごとに代表index（平均 or 最初のdim）
        indices = idx[..., 0]  # [B, T, D]

        # 完全FSQはlossなしなので、vq_lossは0
        vq_loss = torch.tensor(0.0, device=x.device)

        return recon, indices, vq_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # モデル のロード
    whisper_enc = WhisperEncoder(model_name="base", device=device).to(device)
    tokenizer = FSQWhisperTokenizer(
        num_factors=8,
        K=3,
        dim=512,
        vmin=-5.0,
        vmax=5.0
    ).to(device)

    whisper_enc.eval()
    tokenizer.eval()

    # 音声読み込み(とりあえず1ファイルで確認)
    wav_path = "/db/Coco-Nut/Coco-Nut/wav/0/0000.wav"
    wav, sr = torchaudio.load(wav_path)  # [C, T]
    # モノラル化
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # [B, T] にする
    wav = wav.to(device)  # [1, T]

    # Whisper Encoder の出力を得る
    with torch.no_grad():
        enc, frame_lengths = whisper_enc(wav)  # [B, T', dim]

    print("Encoder output shape:", enc.shape)
    print("Frame lengths:", frame_lengths)

    # FSQ Tokenizer に通す
    with torch.no_grad():
        recon, indices, _ = tokenizer(enc)

    print("Reconstructed shape:", recon.shape)
    print("Indices shape:", indices.shape)

    # 確認
    print("Sample indices (first frame):")
    print(indices[0, 0])  # [D]

    print("Min/Max of encoder:", enc.min().item(), enc.max().item())
    print("Min/Max of recon:", recon.min().item(), recon.max().item())