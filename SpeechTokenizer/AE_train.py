from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F
from AE_512to8 import AE_512to8

def train_autoencoder(whisper_enc, dataloader, device="cuda", epochs=10):

    model = AE_512to8().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    whisper_enc.eval()

    for epoch in range(epochs):
        total_loss = 0

        for wav, _ in dataloader:
            wav = wav.to(device)

            with torch.no_grad():
                enc, _ = whisper_enc(wav)  # [B, T, 512]

            B, T, D = enc.shape
            enc_flat = enc.view(-1, D)  # [B*T, 512]

            z, recon = model(enc_flat)

            loss = F.mse_loss(recon, enc_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model