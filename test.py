import os
import sys
sys.path.append("./WavTokenizer")
import torch
from lightning.pytorch.cli import instantiate_class
from WavTokenizer.decoder.pretrained import WavTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAVTOKENIZER_CONFIG_PATH = "./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOKENIZER_MODEL_PATH = "./WavTokenizer/wavtokenizer_large_speech_320_v2.ckpt"
wavtokenizer = WavTokenizer.from_pretrained0802(WAVTOKENIZER_CONFIG_PATH, WAVTOKENIZER_MODEL_PATH)
wavtokenizer = wavtokenizer.to(device)

# wavtokenizer = WavTokenizer.from_hparams()
# wavtokenizer.load_state_dict(torch.load(WAVTOKENIZER_MODEL_PATH, map_location="cpu")["state_dict"])
# wavtokenizer = wavtokenizer.to(device)
# wavtokenizer.eval()