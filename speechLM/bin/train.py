from __future__ import print_function
import argparse
import datetime
import logging

from transformers import AutoModelForCausalLM
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
os.environ["VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record

# 追加インポート
from speechLM.tokenizer.tokenizer import get_qwen_tokenizer
from speechLM.dataset.dataset import AudioLLMDataset
# インポート書き換え
from speechLM.utils.executor import Executor
from speechLM.utils.train_utils import (
    init_distributed,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='Training Qwen-2.5 with Multiple Speech Tokenizers')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', default='llm', help='Model slot in config (fixed to llm for Qwen)')
    parser.add_argument('--model_id', default='Qwen/Qwen2.5-7B-Instruct', help='HuggingFace base LLM ID')

    # Speech Tokenizer選択用引数追加
    parser.add_argument('--tokenizer_type', default=None,
                        choices=['wavtokenizer', 'cosyvoice2', 'encodec'],
                        help='Select Speech Tokenizer method to train')
    parser.add_argument('--token_root_dir', default=None,
                        help='Root directory where extraction .npy files are located')

    parser.add_argument('--config', required=True, help='Config file (e.g. config.yaml)')
    parser.add_argument('--train_data', required=True, help='Train data file (tsv path)')
    parser.add_argument('--cv_data', required=True, help='Cross-validation data file (tsv path)')
    parser.add_argument('--checkpoint', help='Checkpoint model')
    parser.add_argument('--model_dir', required=True, help='Save model root directory')
    parser.add_argument('--tensorboard_dir', default='tensorboard', help='Tensorboard log dir')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl',
                        choices=['nccl', 'gloo'], help='Distributed backend')
    parser.add_argument('--num_workers', default=4, type=int, help='Num of subprocess workers for data loading')
    parser.add_argument('--prefetch', default=100, type=int, help='Prefetch number')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision')
    parser.add_argument('--deepspeed.save_states', dest='save_states', default='model_only',
                        choices=['model_only', 'model+optimizer'], help='Save model/optimizer states')
    parser.add_argument('--timeout', default=60, type=int, help='Timeout for cosyvoice_join.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


# バッチのパディング用データコレーター
class AudioLLMDataCollator:
    def __init__(self, pad_token_id, pad_idx=-100):
        self.pad_token_id = pad_token_id
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # dataset.py 側でリネームしたキー名 ('text_token', 'llm_labels') でバッチを集約
        input_ids_list = [item['text_token'] for item in batch]
        labels_list = [item['llm_labels'] for item in batch]

        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids_list, batch_first=True, padding_value=self.pad_token_id
        # )
        # labels = torch.nn.utils.rnn.pad_sequence(
        #     labels_list, batch_first=True, padding_value=self.pad_idx
        # )
        # attention_mask = input_ids.ne(self.pad_token_id).long()

        # 最大長に合わせてパディング (本家の pad_sequence 処理と同期)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=self.pad_idx
        )
        attention_mask = input_ids.ne(self.pad_token_id).long()

        # Executor の model(input_ids, labels, attention_mask) にインジェクションする辞書
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Qwen-2.5ファファインチューニング用に一律Falseに設定
    gan = False

    # 1. hyperpyyaml によるハイパーパラメータ等のロード
    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)

    # 起動引数で上書き更新
    configs['train_conf'].update(vars(args))

    # configs['train_conf'].update(vars(args)) のすぐ下に追記
    # コマンドライン引数が空(None)の場合、config.yaml の設定値を最優先にする
    if configs['train_conf'].get('tokenizer_type') is None:
        configs['train_conf']['tokenizer_type'] = configs.get('tokenizer_type', 'wavtokenizer')
    if configs['train_conf'].get('token_root_dir') is None:
        configs['train_conf']['token_root_dir'] = configs.get('token_root_dir', './output_tokens')

    print(f"STEP 1: Loaded configuration from {args.config}:")

    # 2. 分散トレーニング環境（DDP/DeepSpeed）の初期化
    init_distributed(args)
    rank = int(os.environ.get('RANK', 0))

    print(f"STEP 2: Distributed environment initialized. Rank: {rank}, Backend: {args.dist_backend}, Engine: {args.train_engine}")

    # 3. tokenizer.py から拡張QwenTokenizerの取得
    logging.info(f"Initializing QwenTokenizer. Target Speech Tokenizer: [{args.tokenizer_type.upper()}]")
    qwen_tokenizer = get_qwen_tokenizer(token_path=args.model_id, skip_special_tokens=False)

    print(f"STEP 3: QwenTokenizer initialized with tokenizer type: {args.tokenizer_type}. Vocabulary size: {len(qwen_tokenizer.tokenizer)}")

    # 4. Dataset & DataLoader の動的構築
    logging.info(f"Setting up AudioLLMDataset for data: {args.train_data}")
    train_dataset = AudioLLMDataset(
        data_list_file=args.train_data,
        token_root_dir=args.token_root_dir,
        tokenizer_type=args.tokenizer_type, # 引数選択されたフォルダを参照
        qwen_tokenizer=qwen_tokenizer,
        mode='train'
    )
    cv_dataset = AudioLLMDataset(
        data_list_file=args.cv_data,
        token_root_dir=args.token_root_dir,
        tokenizer_type=args.tokenizer_type,
        qwen_tokenizer=qwen_tokenizer,
        mode='train',
        shuffle=False,
        partition=False
    )

    collator = AudioLLMDataCollator(pad_token_id=qwen_tokenizer.tokenizer.pad_token_id)

    # IterableDataset 用のカスタム DataLoader 構築
    train_data_loader = DataLoader(
        train_dataset, batch_size=None, pin_memory=args.pin_memory,
        num_workers=args.num_workers, prefetch_factor=args.prefetch
    )
    cv_data_loader = DataLoader(
        cv_dataset, batch_size=None, pin_memory=args.pin_memory,
        num_workers=args.num_workers, prefetch_factor=args.prefetch
    )

    print(f"STEP 4: Datasets and DataLoaders initialized. Train samples: {len(train_dataset)}, CV samples: {len(cv_dataset)}")

    # 5. データローダーをバッチ（コレーター適用）単位でラップして返すヘルパー関数
    # （CosyVoiceの共通Executorへシームレスに渡すために内部でバッチ構築を実行）
    def wrap_loader_with_collator(loader, micro_batch_size):
        batch = []
        for sample in loader:
            batch.append(sample)
            if len(batch) == micro_batch_size:
                yield collator(batch)
                batch = []
        if batch:
            yield collator(batch)

    # 基礎構成のチェックと保存
    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    # 6. ベースとなる Qwen-2.5 モデルの取得
    # model = configs[args.model]
    model = AutoModelForCausalLM.from_pretrained(configs['llm_model_path'])

    # 7. 追加語彙数に合わせたEmbeddingサイズ調整 ＆ LoRA設定
    logging.info(f"Resizing token embeddings to {len(qwen_tokenizer.tokenizer)}")
    model.resize_token_embeddings(len(qwen_tokenizer.tokenizer))

    logging.info("Applying LoRA with Modules-to-Save...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"] # 新規トークン最適化のため必須
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable() # メモリの節約

    # チェックポイント読み込み
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
        else:
            logging.warning('Checkpoint {} does not exist!'.format(args.checkpoint))

    # Cudaデバイスへのディスパッチ
    model = wrap_cuda_model(args, model)

    # 8. CosyVoice2共通のコアパーツ（オプティマイザ・スケジューラー）初期化
    # DeepSpeedを指定した場合は、内部で deepspeed.initialize() が自動連動します
    from speechLM.utils.train_utils import init_optimizer_and_scheduler
    model, optimizer, scheduler, _, _ = init_optimizer_and_scheduler(args, configs, model, gan)
    scheduler.set_step(start_step)

    # 初期ウェイトの永続化
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch

    # 保存先が衝突しないよう、モデル名に手法名を自動マウント
    model_save_tag = f"qwen2.5_{args.tokenizer_type}"
    save_model(model, f"{model_save_tag}_init", info_dict)

    # 9. 訓練の実行マネージャの用意
    executor = Executor(gan=gan)
    executor.step = start_step
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    print(f"Start training loop! Method: {args.tokenizer_type}, Step: {start_step}, Epoch: {start_epoch}")

    # 10. メイントレーニングループ
    # 各データローダーをコレーター付きイテレータに変換して Executor へ供給します
    micro_batch_size = configs['train_conf'].get('batch_size', 1)

    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()

        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))

        # コレーターラップされたデータローダーの生成
        collated_train_loader = wrap_loader_with_collator(train_data_loader, micro_batch_size)
        collated_cv_loader = wrap_loader_with_collator(cv_data_loader, micro_batch_size)

        # CosyVoice2 の標準エポック訓練処理をキック
        executor.train_one_epoc(
            model, optimizer, scheduler,
            collated_train_loader, collated_cv_loader,
            writer, info_dict, scaler, group_join
        )

        # エポック終了時の checkpoint 保存
        if rank == 0:
            save_model(model, f"{model_save_tag}_epoch_{epoch}", info_dict)

        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
