import os
import re
import random
import math
from functools import partial
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

# CosyVoice2 固有の構造クラス（そのまま流用）
class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        self.worker_id = worker_info.id if worker_info is not None else 0
        self.num_workers = worker_info.num_workers if worker_info is not None else 1
        return dict(rank=self.rank, world_size=self.world_size, worker_id=self.worker_id, num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[:self.world_size]
            data = data[self.rank::self.world_size]
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[:self.num_workers]
        data = data[self.worker_id::self.num_workers]
        return data

### cosyvoice2での実装例
# class DataList(IterableDataset):
#     def __init__(self, lists, shuffle=True, partition=True):
#         self.lists = lists
#         self.sampler = DistributedSampler(shuffle, partition)

#     def set_epoch(self, epoch):
#         self.sampler.set_epoch(epoch)

#     def __iter__(self):
#         sampler_info = self.sampler.update()
#         indexes = self.sampler.sample(self.lists)
#         for index in indexes:
#             data = dict(src=self.lists[index])
#             data.update(sampler_info)
#             yield data

# tsvファイルから逐次読み込みに変更
class DataList(IterableDataset):
    def __init__(self, data_list_file):
        self.data_list_file = data_list_file

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        with open(self.data_list_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if line_idx % num_workers != worker_id:
                    continue
                yield { "src": line }


# --- 今回新設・拡張するパイプライン用関数群 ---

def parse_tsv_line(data_src, mode='train'):
    """ TSVから読み込んだ行をパースするプロセッサ """
    for sample in data_src:
        line = sample['src'].strip()
        if not line:
            continue
        parts = line.split('\t')
        speech_path = parts[0]
        text = parts[1] if len(parts) > 1 else ""

        # パスからプレフィックス（000など）とファイルIDを抽出
        match = re.search(r'/([^/]+)/([^/]+)\.(wav|flac)$', speech_path)
        if not match:
            continue

        sample['prefix'] = match.group(1)      # '000'
        sample['file_id'] = match.group(2)     # '000734dcb35d6'
        sample['text'] = text
        yield sample

'''
def load_audio_tokens(data_src, token_root_dir, tokenizer_type, mode='train'):
    """ 該当するnpyファイルからトークンを読み込み、文字列形式に変換するプロセッサ """
    for sample in data_src:
        prefix = sample['prefix']
        file_id = sample['file_id']
        # 例: output_tokens/wavtokenizer/000/000734dcb35d6.npy
        npy_path = os.path.join(token_root_dir, tokenizer_type, prefix, f"{file_id}.npy")

        if not os.path.exists(npy_path):
            print("missing:", npy_path)   # デバッグ用に存在しないファイルをログ出力
            continue  # トークンが存在しない場合はスキップ

        try:
            token_ids = np.load(npy_path)

            ### !!! 要注意 !!! ###
            # 多次元（EnCodecなど[K, T]）の場合は平坦化するか、1レイヤー目だけを使うなど要調整
            # if token_ids.ndim > 1:
            #     token_ids = token_ids[0]  # 一例として最初のコードブックのみを使用

            # wavtokenizer = (1,1,T), cosyvoice2 = (T,)
            token_ids = np.asarray(token_ids).flatten()

            # 数字から文字列形式（例: <audio_tok_45>）に変換
            audio_token_strings = [f"<audio_tok_{int(tid)}>" for tid in token_ids]
            sample['audio_tokens'] = audio_token_strings

            # 高速化のため文字列リストにせず、数値のまま次段へ渡す
            # sample['audio_token_ids'] = token_ids.tolist()
            yield random.sample
        except Exception as e:
            # print("error:", e)  # デバッグ用に例外をキャッチしてログ出力
            continue

def format_llm_prompt(data_src, mode='train'):
    """ トークンとテキストをLLMのプロンプト形式にマージするプロセッサ """
    for sample in data_src:
        text = sample['text']
        audio_tokens_str = "".join(sample['audio_tokens'])

        # Qwen-2.5 (ChatML) 向けのプロンプトテンプレート構成例
        # 音声を入力として受け取り、テキストを書き出すタスクを想定
        prompt = (
            "<|im_start|>user\n"
            f"音声を聞いて書き起こしてください: <|start_of_audio|>{audio_tokens_str}<|end_of_audio|><|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{text}<|im_end|>"
        )

        sample['prompt_text'] = prompt
        yield sample

# Dataset を返す関数
def AudioLLMDataset(data_list_file,
                    token_root_dir,
                    tokenizer_type,
                    mode='train',
                    shuffle=True,
                    partition=True):
    """
    Args:
        data_list_file (str): 'reazonspeech_large.tsv' のパス
        token_root_dir (str): 保存したトークンフォルダのルート ('output_tokens')
        tokenizer_type (str): 'wavtokenizer' / 'cosyvoice2' / 'encodec' のいずれか
    """
    assert mode in ['train', 'inference']
    assert tokenizer_type in ['wavtokenizer', 'cosyvoice2', 'encodec']

    # TSVファイルを一行ずつ読み込む
    # with open(data_list_file, 'r', encoding='utf-8') as f:
    #     lists = [line.strip() for line in f if line.strip()]

    # dataset = DataList(lists, shuffle=shuffle, partition=partition)
    dataset = DataList(data_list_file)

    # 固定のデータ処理パイプラインを定義
    data_pipeline = [
        parse_tsv_line,
        partial(load_audio_tokens, token_root_dir=token_root_dir, tokenizer_type=tokenizer_type),
        format_llm_prompt
    ]

    # 各プロセッサを適用
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)

    return dataset
'''

def load_audio_tokens(data_src, token_root_dir, tokenizer_type, mode='train'):
    """ 該当するnpyファイルからトークンを読み込み、文字列形式に変換するプロセッサ """
    for sample in data_src:
        prefix = sample['prefix']
        file_id = sample['file_id']
        # 例: output_tokens/wavtokenizer/000/000734dcb35d6.npy
        npy_path = os.path.join(token_root_dir, tokenizer_type, prefix, f"{file_id}.npy")

        if not os.path.exists(npy_path):
            print("missing:", npy_path)   # デバッグ用に存在しないファイルをログ出力
            continue  # トークンが存在しない場合はスキップ

        try:
            token_ids = np.load(npy_path)

            ### !!! 要注意 !!! ###
            # 多次元（EnCodecなど[K, T]）の場合は平坦化するか、1レイヤー目だけを使うなど要調整
            # if token_ids.ndim > 1:
            #     token_ids = token_ids[0]  # 一例として最初のコードブックのみを使用

            # wavtokenizer = (1,1,T), cosyvoice2 = (T,)
            token_ids = np.asarray(token_ids).flatten()

            # 数字から文字列形式（例: <audio_tok_45>）に変換
            # audio_token_strings = [f"<audio_tok_{int(tid)}>" for tid in token_ids]
            # sample['audio_tokens'] = audio_token_strings

            # 高速化のため文字列リストにせず、数値のまま次段へ渡す
            sample['audio_token_ids'] = token_ids.tolist()
            yield random.sample
        except Exception as e:
            # print("error:", e)  # デバッグ用に例外をキャッチしてログ出力
            continue

'''
def tokenize_and_format_for_llm(data_src, qwen_tokenizer, mode='train'):
    """ 新設した QwenTokenizer の統合メソッドを呼び出して最終データを作る """
    for sample in data_src:
        audio_tids = sample['audio_token_ids']
        text = sample['text']

        # 新トークナイザーの「チャット・音声エンコーダ」を一撃で呼び出す
        input_ids, labels = qwen_tokenizer.encode_chat(
            audio_tokens_list=audio_tids,
            text_target=text
        )

        sample['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        sample['labels'] = torch.tensor(labels, dtype=torch.long)
        yield sample

def AudioLLMDataset(data_list_file,
                    token_root_dir,
                    tokenizer_type,
                    qwen_tokenizer, # 引数として受け取る
                    mode='train',
                    shuffle=True,
                    partition=True):
    assert mode in ['train', 'inference']

    with open(data_list_file, 'r', encoding='utf-8') as f:
        lists = [line.strip() for line in f if line.strip()]

    dataset = DataList(lists, shuffle=shuffle, partition=partition)

    # パイプラインを構築
    data_pipeline = [
        parse_tsv_line,
        partial(load_audio_tokens, token_root_dir=token_root_dir, tokenizer_type=tokenizer_type),
        partial(tokenize_and_format_for_llm, qwen_tokenizer=qwen_tokenizer) # ここで適用
    ]

    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)

    return dataset
'''

def tokenize_and_format_for_llm(data_src, qwen_tokenizer, mode='train'):
    """
    本家 cosyvoice/dataset/processor.py の tokenize 関数に準拠させた実装
    """
    for sample in data_src:
        assert 'audio_token_ids' in sample
        assert 'text' in sample

        audio_tids = sample['audio_token_ids']
        text = sample['text']

        # トークナイザーからChatML変換されたID列とラベルを取得
        input_ids, labels = qwen_tokenizer.encode_chat(
            audio_tokens_list=audio_tids,
            text_target=text
        )

        # 本家 processor.py の padding が処理しやすいように、
        # LLMに入力する最終的なトークン配列を 'text_token' というキー名で格納
        sample['text_token'] = torch.tensor(input_ids, dtype=torch.long)
        sample['llm_labels'] = torch.tensor(labels, dtype=torch.long)

        yield sample


def AudioLLMDataset(data_list_file,
                    token_root_dir,
                    tokenizer_type,
                    qwen_tokenizer,
                    mode='train',
                    shuffle=True,
                    partition=True):
    assert mode in ['train', 'inference']

    with open(data_list_file, 'r', encoding='utf-8') as f:
        lists = [line.strip() for line in f if line.strip()]

    dataset = DataList(lists, partition=partition)

    # 本家の data_pipeline パイプライン構成を模倣
    data_pipeline = [
        parse_tsv_line,
        partial(load_audio_tokens, token_root_dir=token_root_dir, tokenizer_type=tokenizer_type),
        partial(tokenize_and_format_for_llm, qwen_tokenizer=qwen_tokenizer)
    ]

    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)

    return dataset