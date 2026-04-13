import os
os.environ["VISIBLE_DEVICES"] = "0"
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
from fsq_whisper import FSQWhisperTokenizer, WhisperEncoder
from tqdm import tqdm

# -----------------------------
# Dataset データセットクラス
# -----------------------------
class AudioDataset(Dataset):
    def __init__(self, filelist_txt, sr=16000):
        with open(filelist_txt, "r", encoding="utf-8") as f:
            self.files = [l.strip() for l in f if l.strip()]
        self.sample_rate = sr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])  # [channels, time]

        # モノラル変換
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # [1, time]

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav.squeeze(0)  # [T]

# -----------------------------
# Training Function 学習関数
# -----------------------------
def collate_fn(batch):
    """
    batch: List[Tensor] (each [T])
    """
    lengths = [x.shape[-1] for x in batch]
    max_len = max(lengths)

    padded = []
    for x in batch:
        pad_len = max_len - x.shape[-1]
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        padded.append(x)

    wavs = torch.stack(padded)  # [B, T]
    lengths = torch.tensor(lengths)

    return wavs, lengths

def compute_fsq_stats(indices, D, K):
    """
    評価関数: コードブック利用率とperplexityを計算
    Args:
        indices: [B,T,D] リストのテンソル
        Return 戻り値:
            usage_rate: 使用されたコードの割合（0〜1）
            perplexity: exp(H) の形での多様性指標
    """
    idx_all = torch.cat([x.reshape(-1, D) for x in indices], dim=0)

    # usage
    usage = 0
    for d in range(D):
        usage += len(torch.unique(idx_all[:, d])) / K
    usage /= D

    # perplexity
    per_f = []
    for d in range(D):
        hist = torch.bincount(idx_all[:, d], minlength=K).float()
        prob = hist / (hist.sum() + 1e-10)
        entropy = -(prob * torch.log(prob + 1e-10)).sum()
        per_f.append(torch.exp(entropy))

    perplexity = torch.stack(per_f).mean().item()

    return usage, perplexity

'''
old function (collate_fn 対応前)
def log_and_plot_stats(stats_dict, save_dir):
    """損失・使用率・perplexityをJSON保存&グラフ化"""
    os.makedirs(save_dir, exist_ok=True)
    stats_path = os.path.join(save_dir, "train_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)

    # --- 可視化 ---
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(stats_dict["epoch_loss"], label="Total Loss")
    plt.plot(stats_dict["recon_loss"], label="Recon Loss")
    plt.plot(stats_dict["vq_loss"], label="VQ Loss")
    plt.legend()
    plt.title("Training Losses")

    plt.subplot(2, 1, 2)
    plt.plot(stats_dict["usage_rate"], label="Usage Rate (%)")
    plt.plot(stats_dict["perplexity"], label="Perplexity")
    plt.legend()
    plt.title("Codebook Usage & Diversity")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_curves.png"))
    plt.close()

def train_fsq_tokenizer(
    filelist_txt,           # "../data/coco_path.txt"
    model_name="base",      # WhisperEncoder のモデル名
    batch_size=1,
    epochs=40,
    lr=1e-3,
    D=8,
    K=4,
    dim=512,
    save_dir="./fsq_results",
    device="cuda"
):
    os.makedirs(save_dir, exist_ok=True)
    # 評価用のパラメータ保存
    stats = {
        "epoch_loss": [],
        "recon_loss": [],
        "vq_loss": [],
        "usage_rate": [],
        "perplexity": []
        }
    ds = AudioDataset(filelist_txt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    whisper_enc = WhisperEncoder(model_name=model_name, device=device).to(device)
    whisper_enc.eval()

    tokenizer = FSQWhisperTokenizer(D=D, K=K, dim=dim, decay=0.99, use_ema=True).to(device)
    params = list(tokenizer.parameters())
    if len(params) == 0:
        optimizer = None
        print("No learnable parameters — FSQ will use EMA only.")
    else:
        optimizer = torch.optim.Adam(params, lr=lr)


    # early stopping 用変数
    # best_loss = float("inf")
    # stop_count = 0
    for epoch in range(epochs):
        total_loss = total_recon = total_vq = 0.0
        all_indices = []

        for wav in tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}"):
            wav = wav.to(device)

            with torch.no_grad():
                enc = whisper_enc(wav)  # [T, dim]
            enc = enc.unsqueeze(0)  # [1, T, dim]

            recon, idx, vq_loss = tokenizer(enc)
            recon_loss = F.mse_loss(recon, enc)
            # FSQではoptimizer不要のため削除
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            all_indices.append(idx.detach().cpu())
            total_loss += (recon_loss + vq_loss).item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        # epoch 統計
        avg_loss = total_loss / len(dl)
        avg_rec = total_recon / len(dl)
        avg_vq = total_vq / len(dl)

        stats["epoch_loss"].append(avg_loss)
        stats["recon_loss"].append(avg_rec)
        stats["vq_loss"].append(avg_vq)

        # usage/perplexity
        usage, ppl = compute_fsq_stats(all_indices, D, K)
        stats["usage_rate"].append(usage * 100)
        stats["perplexity"].append(ppl)

        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}, usage={usage*100:.2f}%, ppl={ppl:.2f}")
        log_and_plot_stats(stats, save_dir)

        # エポックごとに保存
        torch.save(tokenizer.state_dict(), os.path.join(
                    save_dir, f"model_epoch{epoch+1}.pt"))
        # np.save(os.path.join(save_dir, f"codebook_epoch{epoch+1}.npy"),
        #                 tokenizer..cpu().numpy())

        with open(os.path.join(save_dir, "train_stats.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "config": {
                        "model_name": model_name,
                        # "codebook_size": codebook_size,
                        "lr": lr,
                        "epochs": epochs
                    },
                    "stats": stats
                }, f, indent=2, ensure_ascii=False)

    print("Training complete. Model & stats saved to:", save_dir)
'''

def train_fsq_tokenizer(
    filelist_txt,               # "../data/coco_path.txt"
    model_name="base",
    batch_size=8,
    epochs=50,
    D=8,
    K=3,
    dim=512,
    save_dir="./fsq_results",
    device="cuda",
    masking=True
):
    os.makedirs(save_dir, exist_ok=True)

    stats = {
        "epoch_loss": [],
        "recon_loss": [],
        "vq_loss": [],
        "usage_rate": [],
        "perplexity": []
    }

    ds = AudioDataset(filelist_txt)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    whisper_enc = WhisperEncoder(model_name=model_name, device=device).to(device)
    whisper_enc.eval()

    tokenizer = FSQWhisperTokenizer(num_factors=D, K=K, dim=dim).to(device)
    params = list(tokenizer.parameters())

    for epoch in range(epochs):
        total_loss = total_recon = total_vq = 0.0
        all_indices = []

        for wav, lengths in tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}"):
            wav = wav.to(device)

            with torch.no_grad():
                enc, frame_lengths = whisper_enc(wav)  # [B, T', dim], [B]

            B, T, _ = enc.shape

            # mask作成（padding除外）
            mask = torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)  # [B, T, 1]

            # FSQに通す、vq_lossはEMA更新のために必要
            recon, idx, vq_loss = tokenizer(enc)

            # recon_loss計算、マスクがある場合は有効な部分のみで計算
            if masking:
                recon_loss = F.mse_loss(
                    recon[mask],
                    enc[mask]
                )
            else:
                recon_loss = F.mse_loss(recon, enc)

            loss = recon_loss + vq_loss

            # EMAのみなので backward不要
            # loss.backward()

            all_indices.append(idx[mask.squeeze(-1)].detach().cpu())

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        # --- epoch stats ---
        avg_loss = total_loss / len(dl)
        avg_rec = total_recon / len(dl)
        avg_vq = total_vq / len(dl)

        stats["epoch_loss"].append(avg_loss)
        stats["recon_loss"].append(avg_rec)
        stats["vq_loss"].append(avg_vq)

        usage, ppl = compute_fsq_stats(all_indices, D, K)
        stats["usage_rate"].append(usage * 100)
        stats["perplexity"].append(ppl)

        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}, usage={usage*100:.2f}%, ppl={ppl:.2f}")

        # --- 保存 ---
        torch.save(tokenizer.state_dict(),
                   os.path.join(save_dir, f"model_epoch{epoch+1}.pt"))

        with open(os.path.join(save_dir, "train_stats.json"), "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "model_name": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size
                },
                "stats": stats
            }, f, indent=2, ensure_ascii=False)

    print("Training complete. Model & stats saved to:", save_dir)

if __name__ == "__main__":
    train_fsq_tokenizer(
        filelist_txt="./dataset/coco_path.txt",
        model_name="base",
        batch_size=1,
        epochs=50,
        D=8,
        K=3,
        dim=512,
        save_dir="./exp/fsq_whisper_base_K3D8",
        device="cuda"
    )
