import os
import _distutils_system_mod
import numpy as np

token_root_dir = "./data/speech_tokens"
tokenizer_type = "cosyvoice2"

""" 該当するnpyファイルからトークンを読み込み、文字列形式に変換するプロセッサ """
prefix = "000"
file_id = "000734dcb35d6"

# 例: output_tokens/cosyvoice2/000/000734dcb35d6.npy
npy_path = os.path.join(token_root_dir, tokenizer_type, prefix, f"{file_id}.npy")
print("npy_path:", npy_path)

# if not os.path.exists(npy_path):
#     continue  # トークンが存在しない場合はスキップ
# デバッグ用に存在しないファイルをログ出力
if not os.path.exists(npy_path):
    print("missing:", npy_path)


try:
    # トークン（整数配列）の読み込み
    token_ids = np.load(npy_path)
    print(token_ids.shape)
    print(token_ids.dtype)
except Exception as e:
    print("error:", e)
