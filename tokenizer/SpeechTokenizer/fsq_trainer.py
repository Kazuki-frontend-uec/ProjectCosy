import torch
import torch.nn as nn
import torch.optim as optim
from whisper_fsq_encoder import WhisperFSQEncoder

import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class DataSets(Dataset):
    def __init__(self, filelist_txt, sample_rate=16000):
        with open(filelist_txt, "r", encoding="utf-8") as f:
            self.files = [l.strip() for l in f if l.strip()]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])  # [channels, time]
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # [1, time]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav.squeeze(0).float()  # [T]

def train_fsq():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperFSQEncoder().to(device)
    ds = DataSets("dataset/coco_path.txt")
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)
    params_to_optimize = list(model.project_in.parameters()) + list(model.project_out.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 10
    best_loss = float("inf")
    for epoch in range(epochs):
        model.project_in.train()
        model.project_out.train()
        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")

        for wav in pbar:
            wav = wav.to(device)
            optimizer.zero_grad()

            orig, recon, _ = model(wav)

            loss = criterion(recon, orig)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        if epoch == 1:
            best_loss = loss.item()
            torch.save({
                'project_in': model.project_in.state_dict(),
                'project_out': model.project_out.state_dict(),
            }, "tokenizer/SpeechTokenizer/exp/wf_4096_best.pth")
        elif loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'project_in': model.project_in.state_dict(),
                'project_out': model.project_out.state_dict(),
            }, "tokenizer/SpeechTokenizer/exp/wf_4096_best.pth")

    torch.save({
        'project_in': model.project_in.state_dict(),
        'project_out': model.project_out.state_dict(),
    }, "tokenizer/SpeechTokenizer/exp/wf_.pth")

if __name__ == "__main__":
    train_fsq()
