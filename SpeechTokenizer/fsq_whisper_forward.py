import torch
import torchaudio
from fsq_whisper import FSQEmbeddingEMA, SpeechTokenizer
from WhisperEncoder import WhisperEncoder

# -----------------------------
# Utility: フォワードしてコード化されたインデックスを得る関数
# -----------------------------
def quantize_audio_paths(wavpaths, fsq_ckpt="fsq_model.pt", device="cuda"):
    fsq = SpeechTokenizer(num_factors=8, K=4, dim=512).to(device)
    fsq.load_state_dict(torch.load(fsq_ckpt, map_location=device))
    fsq.eval()
    whisper_enc = WhisperEncoder(model_name="base", device=device).to(device)
    whisper_enc.eval()

    if isinstance(wavpaths, str):
        # txtファイルとして扱う
        with open(wavpaths, "r", encoding="utf-8") as f:
            wavpaths = [l.strip() for l in f if l.strip()]

    all_indices = []
    for p in wavpaths:
        wav, sr = torchaudio.load(p)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        enc = whisper_enc(wav)  # [T, dim]
        if enc.dim() == 2:
            enc = enc.unsqueeze(0)
        _, indices, _ = fsq(enc)
        all_indices.append(indices.cpu().numpy())
    return all_indices

if __name__ == "__main__":
    filelist_txt = ["/db/Coco-Nut/Coco-Nut/wav/0/0000.wav"]  # 音声ファイルのパスを列挙したテキストファイル
    fsq_ckpt = "/workH/isa/python/prj_cosy/SpeechTokenizer/exp/fsq_practice/model_epoch20.pt"       # 学習済みFSQモデルのチェックポイント
    indices = quantize_audio_paths(filelist_txt, fsq_ckpt)
    print("Quantized indices shape:", [x.shape for x in indices])
