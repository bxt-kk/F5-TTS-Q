import random

from f5_tts.api import F5TTS
from f5_tts.model import DiT
from f5_tts.dataset import ESDDraw
import torch

from tqdm import tqdm


f5tts = F5TTS()
f5tts.ema_model.to(torch.float32)
dit_backbone:DiT = f5tts.ema_model.transformer

nfe_step = 4
sample_k = 3
dit_backbone.nfe_step = nfe_step
root_dir = '/media/kk/Data/dataset/audio/Emotion Speech Dataset'

esd_raw = ESDDraw(root_dir)
table = []
random_idxs = random.sample(range(len(esd_raw)), k=sample_k)
for ix in tqdm(random_idxs, ncols=80):
    ref_file, ref_text, gen_text = esd_raw[ix]
    f5tts.infer(
        ref_file, ref_text, gen_text=gen_text, nfe_step=nfe_step)
    table.append(dit_backbone.trans_dists.clone())
    dit_backbone.trans_dists[:] = 0
torch.save(table, '/tmp/table.pt')
table = torch.load('/tmp/table.pt')
for line in table:
    print(line)
