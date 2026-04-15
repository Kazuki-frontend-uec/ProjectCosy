from SpeechTokenizer.whisper_fsq_encoder import WhisperFSQEncoder
import torch
from transformers import AutoTokenizer
import os
import torchaudio

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
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
    "TTS/B": "TTS/B",
    "TTS/O": "TTS/O",
    "TTS/Q": "TTS/Q",
    "TTS/A": "TTS/A",
    "TTS/CO": "TTS/CO",
    "TTS/CL": "TTS/CL",
    "TTS/H": "TTS/H",
    **{f"TTS/SP{i:02d}": f"TTS/SP{i:02d}" for i in range(1, 14)}
}

# @lru_cache(maxsize=None)
# def get_encoding(name: str = "gpt2", num_languages: int = 99):
#     vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
#     ranks = {
#         base64.b64decode(token): int(rank)
#         for token, rank in (line.split() for line in open(vocab_path) if line)
#     }
#     n_vocab = len(ranks)
#     special_tokens = {}

#     specials = [
#         "<|endoftext|>",
#         "<|startoftranscript|>",
#         *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
#         *[f"<|{audio_event}|>" for audio_event in list(AUDIO_EVENT.keys())],
#         *[f"<|{emotion}|>" for emotion in list(EMOTION.keys())],
#         "<|translate|>",
#         "<|transcribe|>",
#         "<|startoflm|>",
#         "<|startofprev|>",
#         "<|nospeech|>",
#         "<|notimestamps|>",
#         *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],        # register special tokens for ASR
#         *[f"<|{tts}|>" for tts in list(TTS_Vocal_Token.keys())],  # register special tokens for TTS
#         *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
#     ]

#     for token in specials:
#         special_tokens[token] = n_vocab
#         n_vocab += 1

#     return tiktoken.Encoding(
#         name=os.path.basename(vocab_path),
#         explicit_n_vocab=n_vocab,
#         pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
#         mergeable_ranks=ranks,
#         special_tokens=special_tokens,
#     )

# @lru_cache(maxsize=None)
# def get_tokenizer(
#     multilingual: bool,
#     *,
#     num_languages: int = 99,
#     language: Optional[str] = None,
#     task: Optional[str] = None,  # Literal["transcribe", "translate", None]
# ) -> Tokenizer:
#     if language is not None:
#         language = language.lower()
#         if language not in LANGUAGES:
#             if language in TO_LANGUAGE_CODE:
#                 language = TO_LANGUAGE_CODE[language]
#             else:
#                 raise ValueError(f"Unsupported language: {language}")

#     if multilingual:
#         encoding_name = "multilingual_zh_ja_yue_char_del"
#         language = language or "en"
#         task = task or "transcribe"
#     else:
#         encoding_name = "gpt2"
#         language = None
#         task = None

#     encoding = get_encoding(name=encoding_name, num_languages=num_languages)

#     return Tokenizer(
#         encoding=encoding, num_languages=num_languages, language=language, task=task
#     )


class SpeechTokenizer:
    def __init__(self,
                 checkpoint_path="tokenizer/SpeechTokenizer/exp/wf_4096_best.pth",
                 device="cuda",
                 token_offset=160000,
                 sr=16000,
                 levels=[3]*8):
        """
        Args:
            checkpoint_path: 学習した project_in, project_out の重み
            device: 'cuda' or 'cpu'
            token_offset: テキストトークンと被らないように加算する値
        """
        self.device = torch.device(device)
        self.token_offset = token_offset
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

    def preprocess_audio(self, wav_path):
        """音声を読み込んで16kHzモノラルに変換"""
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav)
        return wav.squeeze(0) # [T]

    @torch.no_grad()
    def encode(self, wav_path):
        """音声ファイルをトークンIDのリストに変換"""
        wav = self.preprocess_audio(wav_path).to(self.device)

        # WhisperFSQEncoderのforwardを呼び出し
        # 戻り値: (features, recon_features, indices)
        _, _, indices = self.model(wav)

        # indices は [1, Seq] の形状なので numpy/list に変換
        token_ids = indices.squeeze(0).cpu().numpy()

        # LLM用オフセットの適用
        token_ids = (token_ids + self.token_offset).tolist()

        return token_ids

class QwenTokenizer:
    def __init__(self,
                 llm_model_path="Qwen/Qwen2-7B",
                 speech_tokenizer_path="tokenizer/SpeechTokenizer/exp/wf_4096_best.pth",
                 device="cuda"):

        # 1. テキスト用トークナイザ (Qwen)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

        # 2. 音声用トークナイザ (SpeechTokenizer、自作 or CosyVoice2)
        self.speech_tokenizer = SpeechTokenizer(checkpoint_path=speech_tokenizer_path, device=device, token_offset=0)

        # 3. 特殊トークンの定義
        self.SPECIAL_TOKENS = {
            "so_audio": "<Speech>",           # 音声区間開始
            "eo_audio": "</Speech>",          # 音声区間終了
            "eop": "<|endofprompt|>",         # プロンプトと生成対象の境界
            # "breath": "[breath]",            # 息継ぎ（日本語の自然さに重要）
            # "happy": "<|HAPPY|>",            # 感情制御系
            # "sad": "<|SAD|>",
            # "neutral": "<|NEUTRAL|>"
        }
        # 4. 音声トークンリストの作成 (<speech_0> ~ <speech_6560>)
        self.speech_tokens = [f"<speech_{i}>" for i in range(6561)]
        print(f"Total speech tokens: {len(self.speech_tokens)}")
        print(f"Sample speech tokens: {self.speech_tokens[:5]} ... {self.speech_tokens[-5:]}")


        # トークナイザに特殊トークンを追加
        special_list = list(self.SPECIAL_TOKENS.values()) + self.speech_tokens
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': special_list})
        # special_list = list(self.SPECIAL_TOKENS.values())
        # self.llm_tokenizer.add_special_tokens({'additional_special_tokens': special_list})

       # 3. IDへの変換（高速化のため保持）
        self.so_audio_id = self.llm_tokenizer.convert_tokens_to_ids("<Speech>")
        self.eo_audio_id = self.llm_tokenizer.convert_tokens_to_ids("</Speech>")
        self.eop_id = self.llm_tokenizer.convert_tokens_to_ids("<|endofprompt|>")
        self.speech_token_ids = self.llm_tokenizer.convert_tokens_to_ids(self.speech_tokens)

    def encode(self, text, wav_path):
        # 1. テキスト部分
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = self.llm_tokenizer.encode(prompt, add_special_tokens=False)

        # 2. 音声部分 (SpeechTokenizer側で .tolist() 済みであることを前提)
        raw_audio_indices = self.speech_tokenizer.encode(wav_path) # 0~6560のリスト
        audio_ids = [self.speech_token_ids[idx] for idx in raw_audio_indices]
        # audio_ids = self.speech_tokenizer.encode(wav_path)

        # もし SpeechTokenizer.encode がリストを返さない場合の保険
        if torch.is_tensor(audio_ids):
            audio_ids = audio_ids.flatten().tolist()

        # 3. 結合
        # 全てが int であることを保証
        input_ids = (
            text_ids +
            [self.eop_id, self.so_audio_id] +
            audio_ids +
            [self.eo_audio_id, self.llm_tokenizer.eos_token_id]
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
            if i >= 160000:
                # 音声トークンはIDとして表示
                # tokens.append(f"[A:{i}]")
                continue
            else:
                # テキストトークンは1つずつデコード（特殊トークンも含む）
                tokens.append(self.llm_tokenizer.decode([i]))

        return "".join(tokens)

# 使用
if __name__ == "__main__":
    # ckpt = "tokenizer/SpeechTokenizer/exp/wf_4096_best.pth"
    # tknz = SpeechTokenizer(checkpoint_path=ckpt)

    # # テスト用の音声ファイルを指定
    # test_wav = "/db/Coco-Nut/Coco-Nut/wav/0/0000.wav"
    # if os.path.exists(test_wav):
    #     tokens = tknz.encode(test_wav)
    #     print(f"抽出されたトークン数: {len(tokens)}")
    #     print(f"先頭10トークン (Offset適用済み): {tokens[:10]}")
    # else:
    #     print("テスト用音声ファイルが見つかりません。")


    it = QwenTokenizer(llm_model_path="Qwen/Qwen2-7B-Instruct")

    sample_text = "里羊が3匹"
    sample_wav = "/db/Coco-Nut/Coco-Nut/wav/0/0001.wav"

    input_ids = it.encode(sample_text, sample_wav)

    print(f"Total Sequence Length: {len(input_ids)}")
    print(f"First 5 IDs: {input_ids[:5]}")  # SO_AUDIO + Audio IDs...
    print(f"Last 5 IDs: {input_ids[-5:]}")   # ...Text IDs + EOS

    # print("\n--- Full Structure Decode ---")
    # print(it.decode(input_ids))
