import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import whisper
from whisper.audio import pad_or_trim
import matplotlib.pyplot as plt
from scipy.stats import entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# =========================================================
#  Dataset
# =========================================================
class AudioDataset(Dataset):
    def __init__(self, filelist_txt, sample_rate=16000):
        with open(filelist_txt, "r", encoding="utf-8") as f:
            self.files = [l.strip() for l in f if l.strip()]
        self.sample_rate = sample_rate

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


# =========================================================
#  Whisper Encoder 抽出ラッパ
# =========================================================
class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        # load_model に device を渡しておくと model が自動でデバイスに乗る
        self.model = whisper.load_model(model_name, device=device)
        self.encoder = self.model.encoder
        self.mel_fn = whisper.log_mel_spectrogram

    @torch.no_grad()
    def forward(self, wav):
        # wav: tensor [T] あるいは [1, T] など
        # 1) モノラル化 & squeeze
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0)
        elif wav.dim() == 2:
            wav = wav.squeeze(0)

        # 2) CPU上の numpy にして pad_or_trim (30s相当に揃える)
        #    pad_or_trim expects 1D numpy array
        wav_np = wav.detach().cpu().numpy()
        wav_np = pad_or_trim(wav_np)          # length = 30 * 16000

        # 3) back to torch tensor on device
        wav_trim = torch.from_numpy(wav_np).float().to(self.device)

        # 4) mel spectrogram (on device)
        mel = self.mel_fn(wav_trim).to(self.device)  # shape [80, 3000]

        # 5) ensure batch dim: [1, 80, 3000]
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        # 6) encoder forward (already on device)
        emb = self.encoder(mel)  # whisper encoder expects [batch, 80, 3000]
        return emb.squeeze(0)    # [frames, dim]

# =========================================================
#  VQ Embedding (EMA 更新)
# =========================================================
class VQEmbeddingEMA(nn.Module):
    def __init__(self, num_embeddings=2048, embedding_dim=512, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e):
        flat = z_e.view(-1, self.embedding_dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        idx = torch.argmin(dist, dim=1)
        z_q = self.embedding[idx].view_as(z_e)

        if self.training:
            enc_onehot = F.one_hot(idx, self.num_embeddings).type_as(flat)
            cluster_size = enc_onehot.sum(0)
            embed_sum = enc_onehot.t() @ flat
            self.cluster_size.data.mul_(self.decay).add_(
                cluster_size, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.num_embeddings * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        z_q = z_e + (z_q - z_e).detach()
        commit_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + 0.25 * commit_loss
        return z_q, idx, vq_loss

# =========================================================
#  Speech Tokenizer モデル
# =========================================================
class SpeechTokenizer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, codebook_size=2048):
        super().__init__()
        self.proj_down = nn.Linear(input_dim, hidden_dim)
        self.vq = VQEmbeddingEMA(
            num_embeddings=codebook_size, embedding_dim=hidden_dim)
        self.proj_up = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = self.proj_down(x)
        z_q, idx, vq_loss = self.vq(h)
        recon = self.proj_up(z_q)
        recon_loss = F.mse_loss(recon, x)
        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss, vq_loss, idx

# =========================================================
#  評価関数
# =========================================================
def compute_codebook_stats(indices, num_embeddings):
    """コードブック利用率とperplexityを計算"""
    indices = torch.cat(indices).detach().cpu()
    usage = torch.unique(indices)
    usage_rate = len(usage) / num_embeddings

    counts = torch.bincount(indices, minlength=num_embeddings).float()
    p = counts / counts.sum()
    p = p[p > 0]
    perplexity = float(np.exp(entropy(p.numpy())))

    return usage_rate, perplexity

def log_and_plot_stats(stats_dict, save_dir):
    """損失・使用率・perplexityをJSON保存＆グラフ化"""
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

def evaluate_codebook_usage(model_path, codebook_size, filelist_txt, device="cuda"):
    """保存済みモデルをロードし、コードブック利用率とperplexityを計算"""
    # モデル読み込み
    tokenizer = SpeechTokenizer(
        input_dim=512, hidden_dim=512, codebook_size=codebook_size
    ).to(device)
    tokenizer.load_state_dict(torch.load(model_path, map_location=device))
    tokenizer.eval()

    # データセット準備
    ds = AudioDataset(filelist_txt)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    whisper_enc = WhisperEncoder("base", device=device).to(device)
    whisper_enc.eval()

    all_indices = []
    for wav in tqdm(dl, desc="Evaluating"):
        wav = wav.to(device)
        with torch.no_grad():
            emb = whisper_enc(wav)
            emb = emb.unsqueeze(0)
            _, _, _, idx = tokenizer(emb)
            all_indices.append(idx.flatten())

    usage_rate, perplexity = compute_codebook_stats(all_indices, codebook_size)
    print(f"Usage Rate: {usage_rate*100:.2f}%  |  Perplexity: {perplexity:.2f}")
    return usage_rate, perplexity

# =========================================================
#  学習関数
# =========================================================
def train_vq_tokenizer(
    filelist_txt,
    model_name="medium",
    codebook_size=2048,
    batch_size=1,
    epochs=10,
    lr=1e-4,
    save_dir="./vq_results",
    device="cuda",
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
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    whisper_enc = WhisperEncoder(model_name=model_name, device=device).to(device)
    whisper_enc.eval()

    tokenizer = SpeechTokenizer(
        input_dim=512, hidden_dim=512, codebook_size=codebook_size).to(device)
    opt = torch.optim.Adam(tokenizer.parameters(), lr=lr)

    best_loss = float("inf")
    stop_count = 0
    for epoch in range(epochs):
        tokenizer.train()
        total_loss, rec_sum, vq_sum = 0, 0, 0
        all_indices = []
        for wav in tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}"):
            wav = wav.to(device)
            with torch.no_grad():
                enc_out = whisper_enc(wav)
            enc_out = enc_out.unsqueeze(0)  # [B=1,T,D]

            loss, rec, vq, idx = tokenizer(enc_out)
            opt.zero_grad()
            loss.backward()
            opt.step()

            all_indices.append(idx.flatten())
            total_loss += loss.item()
            rec_sum += rec.item()
            vq_sum += vq.item()

        avg_loss = total_loss / len(dl)
        avg_rec = rec_sum / len(dl)
        avg_vq = vq_sum / len(dl)
        stats["epoch_loss"].append(avg_loss)
        stats["recon_loss"].append(avg_rec)
        stats["vq_loss"].append(avg_vq)

        # === コードブック利用率とperplexity計算 ===
        usage_rate, perplexity = compute_codebook_stats(all_indices, tokenizer.vq.num_embeddings)
        stats["usage_rate"].append(usage_rate * 100)
        stats["perplexity"].append(perplexity)
        print(
            f"Epoch {epoch+1}: total={avg_loss:.4f}, recon={avg_rec:.4f}, vq={avg_vq:.4f}, Usage={usage_rate*100:.1f}%, Perplexity={perplexity:.1f}")
        log_and_plot_stats(stats, save_dir)

        # === エポックごとに統計保存 ===
        with open(os.path.join(save_dir, "train_stats.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "config": {
                        "model_name": model_name,
                        "codebook_size": codebook_size,
                        "lr": lr,
                        "epochs": epochs
                    },
                    "stats": stats
                }, f, indent=2, ensure_ascii=False)

        # === 5エポックごとにモデル保存 ===
        if epoch == 0 or (epoch+1) % 5 == 0:
            torch.save(tokenizer.state_dict(), os.path.join(
                save_dir, f"model_epoch{epoch+1}.pt"))
            torch.save(opt.state_dict(), os.path.join(
                save_dir, f"optimizer_epoch{epoch+1}.pt"))
            np.save(os.path.join(save_dir, f"codebook_epoch{epoch+1}.npy"),
                    tokenizer.vq.embedding.detach().cpu().numpy())

        if epoch ==0:
            best_loss = avg_loss
        elif best_loss - avg_loss > 1e-3:
            stop_count = 0
            best_loss = avg_loss
            # === ベストモデル保存 ===
            torch.save(tokenizer.state_dict(), os.path.join(
                save_dir, f"best_model.pt"))
            torch.save(opt.state_dict(), os.path.join(
                save_dir, f"best_optimizer.pt"))
            np.save(os.path.join(save_dir, f"best_codebook.npy"),
                    tokenizer.vq.embedding.detach().cpu().numpy())
        elif best_loss - avg_loss > 0:
            stop_count += 1
            best_loss = avg_loss
            # === ベストモデル保存 ===
            torch.save(tokenizer.state_dict(), os.path.join(
                save_dir, f"best_model.pt"))
            torch.save(opt.state_dict(), os.path.join(
                save_dir, f"best_optimizer.pt"))
            np.save(os.path.join(save_dir, f"best_codebook.npy"),
                    tokenizer.vq.embedding.detach().cpu().numpy())
            if stop_count >= 3:
                print("Early stopping triggered.")
                break
        else:
            stop_count +=1

    print("Training complete. Model & stats saved to:", save_dir)



# =========================================================
#  実行例
# =========================================================
if __name__ == "__main__":
    train_vq_tokenizer(
        filelist_txt="./data/coco_path.txt",
        model_name="base",
        codebook_size=512,
        batch_size=1,
        epochs=100,
        lr=1e-4,
        save_dir="./exp/vq_2048_ep100_2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # evaluate_codebook_usage(model_path="./vq_tokenizer_out/model_epoch10.pt",
    #                         codebook_size=512,
    #                         filelist_txt="../data/coco_path.txt",
    #                         device="cuda" if torch.cuda.is_available() else "cpu")
