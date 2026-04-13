import json
import torch
from tqdm import tqdm

def pre_tokenize_dataset(input_jsonl, output_jsonl, speech_tokenizer):
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in):
            data = json.loads(line)
            wav_path = data['wav_path']

            try:
                # 音声をトークンID（リスト）に変換
                # すでに token_offset が加算された状態
                audio_ids = speech_tokenizer.encode(wav_path)

                # 保存用データ
                entry = {
                    "text": data['text'],
                    "audio_tokens": audio_ids
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

# 実行
# pre_tokenize_dataset("raw_data.jsonl", "processed_data.jsonl", my_speech_tokenizer)