import base64
import os
os.environ["visible_devices"] = "0"
from functools import lru_cache
from typing import Optional
import torch
from transformers import AutoTokenizer
from whisper.tokenizer import Tokenizer
from SpeechTokenizer.whisper_fsq_encoder import WhisperFSQEncoder
import tiktoken
import numpy as np
import onnxruntime as ort
import torchaudio


# TOKENS = {
#     # 制御
#     "<|eos|>", "<|bos|>",
#     # 言語
#     "<|ja|>", "<|en|>",
#     # タスク
#     "<|tts|>", "<|asr|>",
#     # 音声イベント
#     "<|laugh|>", "<|breath|>",
#     # 感情
#     "<|happy|>", "<|sad|>", "<|neutral|>",
#     # 時間（超重要）
#     "<|0.00|>", "<|0.02|>", ..., "<|30.00|>",
#     # 音声トークン
#     "<|audio_0|>" ... "<|audio_4095|>"
# }

LANGUAGES = {
    "en": "english",
    "ja": "japanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
}

AUDIO_EVENT = {
    "ASR": "ASR",
    "AED": "AED",
    "SER": "SER",
    "Speech": "Speech",
    "/Speech": "/Speech",
    "BGM": "BGM",
    "/BGM": "/BGM",
    "Laughter": "Laughter",
    "/Laughter": "/Laughter",
    "Applause": "Applause",
    "/Applause": "/Applause",
}

EMOTION = {
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "NEUTRAL": "NEUTRAL",
}

TTS_Vocal_Token = {
    "TTS/B": "TTS/B",   # Begin（開始・ベース音声）
    "TTS/O": "TTS/O",   # Original / Ordinary（通常発話）
    "TTS/Q": "TTS/Q",   # Question（疑問調）
    "TTS/A": "TTS/A",   # Accent / Affect（強調）
    "TTS/CO": "TTS/CO", # Continuation（継続）
    "TTS/CL": "TTS/CL", # Close（終了・区切り）
    "TTS/H": "TTS/H",   # High / Happy（高揚・高ピッチ）
    **{f"TTS/SP{i:02d}": f"TTS/SP{i:02d}" for i in range(1, 14)}    # 意味合いは後付け可能
}

@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        *[f"<|{audio_event}|>" for audio_event in list(AUDIO_EVENT.keys())],
        *[f"<|{emotion}|>" for emotion in list(EMOTION.keys())],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],        # register special tokens for ASR
        *[f"<|{tts}|>" for tts in list(TTS_Vocal_Token.keys())],  # register special tokens for TTS
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )

@lru_cache(maxsize=None)
def get_tokenizer(
        multilingual: bool,
        *,
        num_languages: int = 99,
        language: Optional[str] = None,
        task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    ) -> Tokenizer:
        if language is not None:
            language = language.lower()
            if language not in LANGUAGES:
                if language in TO_LANGUAGE_CODE:
                    language = TO_LANGUAGE_CODE[language]
                else:
                    raise ValueError(f"Unsupported language: {language}")

        if multilingual:
            encoding_name = "multilingual_zh_ja_yue_char_del"
            language = language or "en"
            task = task or "transcribe"
        else:
            encoding_name = "gpt2"
            language = None
            task = None

        encoding = get_encoding(name=encoding_name, num_languages=num_languages)

        return Tokenizer(
            encoding=encoding, num_languages=num_languages, language=language, task=task
        )

class QwenTokenizer():
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text

@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    token_path: str,
    skip_special_tokens: bool
) -> QwenTokenizer:
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)

class SpeechTokenizer:
    def __init__(self,
                 checkpoint_path="tokenizer/SpeechTokenizer/exp/wf_4096_best.pth",
                 device="cuda",
                 sr=16000,
                 levels=[3]*8):
        """
        Args:
            checkpoint_path: 学習した project_in, project_out の重み
            device: 'cuda' or 'cpu'
            token_offset: テキストトークンと被らないように加算する値
        """
        self.device = torch.device(device)
        self.sr = sr
        self.num_codebooks = levels.__len__() ** max(levels)  # 3^8 = 6561
        # 1. モデルの初期化 (FSQの構成 [3]*8 は学習時と合わせる)
        self.model = WhisperFSQEncoder(levels=levels, device=device)

        # 2. 学習済み重み (Projector) のロード
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}...")
            # map_location を指定して CPU/GPU 両対応
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.project_in.load_state_dict(checkpoint['project_in'])
            self.model.project_out.load_state_dict(checkpoint['project_out'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, speech_path):
        """音声ファイルをトークンIDのリストに変換"""
        wav, sr = torchaudio.load(speech_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav)
        wav = wav.squeeze(0)  # [T]
        
        # WhisperFSQEncoderのforwardを呼び出し
        # 戻り値: (features, recon_features, indices)
        _, _, indices = self.model(wav)

        token_ids = indices.squeeze(0).cpu().tolist()

        return token_ids

class CosyTokenizer:
    def __init__(self, onnx_path: str, sr=16000):
        self.session = ort.InferenceSession(onnx_path, providers='CUDAExecutionProvider')
        self.sr = sr

    def encode(self, speech_path):
        """音声を読み込んでトークンIDのリストに変換"""
        wav, sr = torchaudio.load(speech_path)
        # 16kHzにリサンプル
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # モノラル化
        wav = wav.mean(dim=0, keepdim=True)  # [1, T]
        
        # log-mel spectrogram作成
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=400,         # 25ms
            hop_length=320,    # 10ms=>160 samples, 20ms=>320 samples
            win_length=400,
            n_mels=128,
            f_min=0,
            f_max=8000,
            center=True,
            power=2.0
        )

        mel = mel_transform(wav)            # [1, 128, T]
        log_mel = torch.log(mel + 1e-6)     # [1, 128, T]


        # length作成
        # データ型は int32 で、フレーム数は T = wav_length // (hop_length = 160)
        feats = log_mel.numpy().astype(np.float32)
        feats_length = np.array([feats.shape[1]], dtype=np.int32)

        multiple = 32
        T = feats.shape[2]
        pad_len = (multiple - (T % multiple)) % multiple

        if pad_len > 0:
            pad = np.zeros((feats.shape[0], feats.shape[1], pad_len), dtype=feats.dtype)
            feats = np.concatenate([feats, pad], axis=2)

        outputs = self.session.run(
            None,
            {
                "feats": feats,
                "feats_length": feats_length
            }
        )

        tokens = outputs[0]

        return tokens  # [B, T]

class UnifiedTokenizer:
    def __init__(self, text_tokenizer_path: str, speech_tokenizer):
        # 1. トークナイザ定義
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        self.speech_tokenizer = speech_tokenizer

        # 2. 特殊トークンの定義(仮)
        self.SPECIAL_TOKENS = {
            "so_audio": "<Speech>",           # 音声区間開始
            "eo_audio": "</Speech>",          # 音声区間終了
            "eop": "<|endofprompt|>",         # プロンプトと生成対象の境界
            # "breath": "[breath]",            # 息継ぎ（日本語の自然さに重要）
            # "happy": "<|HAPPY|>",            # 感情制御系
            # "sad": "<|SAD|>",
            # "neutral": "<|NEUTRAL|>"
        }

        # 3. 音声トークンリストの作成 (<speech_0> ~ <speech_6560>)
        self.speech_tokens = [f"<speech_{i}>" for i in range(6561)]
        print(f"Total speech tokens: {len(self.speech_tokens)}")
        print(f"Sample speech tokens: {self.speech_tokens[:5]} ... {self.speech_tokens[-5:]}")
        special_list = list(self.SPECIAL_TOKENS.values()) + self.speech_tokens
        self.text_tokenizer.add_special_tokens({'additional_special_tokens': special_list})
        
        # 4. IDへの変換（高速化のため保持）
        self.so_audio_id = self.text_tokenizer.convert_tokens_to_ids("<Speech>")
        self.eo_audio_id = self.text_tokenizer.convert_tokens_to_ids("</Speech>")
        self.eop_id = self.text_tokenizer.convert_tokens_to_ids("<|endofprompt|>")
        self.speech_token_ids = self.text_tokenizer.convert_tokens_to_ids(self.speech_tokens)

    def encode(self, text, speech_path):
        # 1. user部分
        user_prompt = f"<|im_start|>user\n{text}\n<|im_end|>\n"
        user_ids = self.text_tokenizer.encode(user_prompt, add_special_tokens=False)

        # 2. assistant開始
        assistant_prompt = f"<|im_start|>assistant\n"
        assistant_ids = self.text_tokenizer.encode(assistant_prompt, add_special_tokens=False)

        # 3. 音声トークン
        raw_audio_indices = self.speech_tokenizer.encode(speech_path)  # 0~6560
        audio_ids = [self.speech_token_ids[idx] for idx in raw_audio_indices]

        # 念のため
        if torch.is_tensor(audio_ids):
            audio_ids = audio_ids.flatten().tolist()

        # 4. 結合（構造を明示）
        input_ids = (
            user_ids +
            assistant_ids +
            [self.eop_id, self.so_audio_id] +
            audio_ids +
            [self.eo_audio_id] +
            self.text_tokenizer.encode("<|im_end|>", add_special_tokens=False)
        )

        return torch.tensor(input_ids, dtype=torch.long)

    def decode(self, input_ids):
        """
        デバッグ用：音声トークンを [AUDIO_ID] という形式で可視化しつつデコード
        """
        if torch.is_tensor(input_ids):
            input_ids = input_ids.flatten().tolist()

        tokens = []
        for i in input_ids:
            # if i >= 160000:
            if i in self.speech_token_ids:
                # 音声トークンはIDとして表示
                # tokens.append(f"[A:{i}]")
                continue
            else:
                # テキストトークンは1つずつデコード（特殊トークンも含む）
                tokens.append(self.text_tokenizer.decode([i]))

        return "".join(tokens)

# トークン配列メモ
# <|im_start|>user
# テキスト
# <|im_end|>
# <|im_start|>assistant
# <|endofprompt|>
# <Speech>
# [speech tokens...]
# </Speech>
# <|im_end|>

if __name__ == "__main__":
    checkpoint_path="tokenizer/SpeechTokenizer/exp/wf_4096_best.pth"
    speech_tokenizer = SpeechTokenizer(checkpoint_path=checkpoint_path, device="cuda")
    tokenizer = UnifiedTokenizer(
        text_tokenizer_path="tokenizer/config",
        speech_tokenizer=speech_tokenizer
    )
    sample_text = "里羊が3匹"
    sample_wav = "/db/Coco-Nut/Coco-Nut/wav/0/0001.wav"

    input_ids = tokenizer.encode(sample_text, sample_wav)

    print(f"Total Sequence Length: {len(input_ids)}")
    print(f"First 5 IDs: {input_ids[:5]}")  # SO_AUDIO + Audio IDs...
    print(f"Last 5 IDs: {input_ids[-5:]}")   # ...Text IDs + EOS
