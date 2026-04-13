# https://zenn.dev/sunwood_ai_labs/articles/openai-gpt-oss-20b-fine-tuning
# https://zenn.dev/prgckwb/articles/gpt-oss-finetuning-unsloth

from unsloth import FastLanguageModel
import torch

max_seq_length = 4096
dtype = None

# 4ビット量子化済みモデル（高速ダウンロード＆OOM回避）
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # bitsandbytes 4ビット量子化
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b",                    # MXFP4フォーマット
    "unsloth/gpt-oss-120b",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = dtype,  # 自動検出
    max_seq_length = max_seq_length,
    load_in_4bit = True,  # 4ビット量子化でメモリ削減
    full_finetuning = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,  # ランク（8, 16, 32, 64, 128から選択）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,  # 0が最適化済み
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # 30%少ないVRAM使用
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

from transformers import TextStreamer

messages = [
    {"role": "user", "content": "5・7・5で冬をテーマにした俳句を日本語で作ってください。"},
]

# 推論努力レベルを設定（low/medium/high）
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "medium",  # ここで設定！
).to(model.device)

_ = model.generate(**inputs, max_new_tokens = 1024, streamer = TextStreamer(tokenizer))
