import os
import re
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
import whisper

# 1. GPU / 環境設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# パス設定
# TSV_PATH = "dataset/reazonspeech_test.tsv"
TSV_PATH = "dataset/reazonspeech_large.tsv"   # 「/pathtowav.wav aligment」のセットデータ
OUTPUT_DIR = Path("dataset/speech_tokens")
COSYVOICE_ONNX_PATH = "CosyVoice_pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx"
WAVTOKENIZER_CONFIG_PATH = "./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOKENIZER_MODEL_PATH = "./WavTokenizer/wavtokenizer_large_speech_320_v2.ckpt"


# 2. 各種モデルの初期化
print("Initializing models...")

# (A) CosyVoice2 (ONNX)
# GPUが使える環境なら CUDAExecutionProvider を優先
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
option = ort.SessionOptions()
option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 1
cosy_session = ort.InferenceSession(COSYVOICE_ONNX_PATH, sess_options=option, providers=providers)

# (B) WavTokenizer
sys.path.insert(0, "./WavTokenizer")
from WavTokenizer.encoder.utils import convert_audio
from WavTokenizer.decoder.pretrained import WavTokenizer
try:
    wavtokenizer = WavTokenizer.from_pretrained0802(WAVTOKENIZER_CONFIG_PATH, WAVTOKENIZER_MODEL_PATH)
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()
except Exception:
    print("WavTokenizerのHF自動ロードに失敗しました。ローカルスクリプトでの読み込みに切り替えてください。")
    wavtokenizer = None
sys.path.remove("./WavTokenizer")

# wav, sr = torchaudio.load(audio_path)
# wav = convert_audio(wav, sr, 24000, 1)
# bandwidth_id = torch.tensor([0])
# wav=wav.to(device)
# _,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
# print(discrete_code)

"""
# (C) EnCodec
from encodec import EncodecModel
from encodec.utils import convert_audio
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0) # 6.0 kbps 等、用途に合わせて帯域を設定
encodec_model = encodec_model.to(device)
encodec_model.eval()
"""

# 3. 各モデルのトークン抽出用ヘルパー関数

def extract_cosyvoice2(wav, sr):
    # 16kHzにリサンプル & モノラル化
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0, keepdim=True)
    feat = whisper.log_mel_spectrogram(wav, n_mels=128)
    speech_token = cosy_session.run(None, {cosy_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                              cosy_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
    return speech_token

def extract_wavtokenizer(wav, sr):
    if wavtokenizer is None:
        return np.array([])

    if sr != 24000:
        wav = convert_audio(wav, sr, 24000, 1)

    with torch.no_grad():
        bandwidth_id = torch.tensor([0])
        wav=wav.to(device)
        _,tokens= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    return tokens.cpu().numpy().flatten()

"""
def extract_encodec(wav, sr):
    # EnCodec (24kHzモデル)
    wav_mono = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
    wav_mono = wav_mono.unsqueeze(0).to(device) # バッチ次元追加 [1, 1, T]

    with torch.no_grad():
        frames = encodec_model.encode(wav_mono)
        # framesは [(codes, scale)] のリスト。codesの形状は [B, K, T] (Kはコードブック数)
        tokens = torch.cat([encoded[0] for encoded in frames], dim=-1)
    return tokens.squeeze(0).cpu().numpy() # [K, T] または一元化
"""

# 4. メイン処理：TSVの読み込みと保存
print("Starting token extraction...")

with open(TSV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # タブ区切りで音声パスとテキストを分離
        parts = line.split("\t")
        speech_path = parts[0]

        if not os.path.exists(speech_path):
            print(f"File not found: {speech_path}")
            continue

        # パスからプレフィックス(例: '000')とファイルID(例: '000734dcb35d6')を抽出
        # パス例: /workN/isa/reazonspeech/wav/000/000734dcb35d6.wav
        match = re.search(r'/([^/]+)/([^/]+)\.(wav|flac)$', speech_path)
        if not match:
            print(f"Skipping invalid path pattern: {speech_path}")
            continue

        prefix = match.group(1) # '000'
        file_id = match.group(2) # '000734dcb35d6'

        try:
            # 音声のロード
            wav, sr = torchaudio.load(speech_path)

            print(f"Processing {speech_path} (SR: {sr}, Shape: {wav.shape})")

            # 各モデルのトークン抽出
            tokens_cosy = extract_cosyvoice2(wav, sr)
            tokens_wavtok = extract_wavtokenizer(wav, sr)
            # tokens_encodec = extract_encodec(wav, sr)

            # 各モデルの出力先ディレクトリ作成と保存
            for model_name, tokens in [
                ("cosyvoice2", tokens_cosy),
                ("wavtokenizer", tokens_wavtok),
                # ("encodec", tokens_encodec)
            ]:
                save_dir = OUTPUT_DIR / model_name / prefix
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / f"{file_id}.npy"
                np.save(save_path, tokens)

        except Exception as e:
            print(f"Error processing {speech_path}: {e}")

print("All tokens have been extracted and saved successfully!")
