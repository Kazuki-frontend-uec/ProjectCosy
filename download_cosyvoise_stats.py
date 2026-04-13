from modelscope import snapshot_download
import os

if not os.path.exists('pretrained_models'):
    os.makedirs('pretrained_models')
if not os.path.exists('pretrained_models/CosyVoice2-0.5B'):
    os.makedirs('pretrained_models/CosyVoice2-0.5B')
# if not os.path.exists('pretrained_models/CosyVoice-300M'):
#     os.makedirs('pretrained_models/CosyVoice-300M')
# if not os.path.exists('pretrained_models/CosyVoice-300M-25Hz'):
#     os.makedirs('pretrained_models/CosyVoice-300M-25Hz')
# if not os.path.exists('pretrained_models/CosyVoice-300M-SFT'):
#     os.makedirs('pretrained_models/CosyVoice-300M-SFT')
# if not os.path.exists('pretrained_models/CosyVoice-300M-Instruct'):
#     os.makedirs('pretrained_models/CosyVoice-300M-Instruct')
# if not os.path.exists('pretrained_models/CosyVoice-ttsfrd'):
#     os.makedirs('pretrained_models/CosyVoice-ttsfrd')

snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
# snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
# snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
# snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
# snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
