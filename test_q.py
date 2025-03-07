from f5_tts.api import F5TTS
from f5_tts.modules import quant_transformer_blocks
from f5_tts.model import DiT
from x_transformers.x_transformers import RotaryEmbedding
import torch


f5tts = F5TTS()
device = f5tts.device

f5tts.ema_model.to(torch.float32)
dit:DiT = f5tts.ema_model.transformer
status = [2, 2, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
blocks_q, status_q = quant_transformer_blocks(dit.transformer_blocks, status)
blocks_q.to(device)
blocks_q_state = torch.load('./ckpts/checkpoint2.pt')['model']
blocks_q.load_state_dict(blocks_q_state)

time_embed_trans = torch.load('./ckpts/time_embed_trans.pt')['W0'].to(device)
rotary_embed_q = RotaryEmbedding(32).to(device)

USE_DITQ = True # 使用DITQ

if USE_DITQ:
    dit.set_q_params(blocks_q, time_embed_trans, rotary_embed_q, status_q)

f5tts.infer(
    './src/f5_tts/infer/examples/basic/basic_ref_zh.wav', # 音频在这里替换
    ref_text='对，这就是我，万人敬仰的太乙真人.',
    gen_text="I don't really care what you call me.",
    file_wave='tests/infer_output.wav', nfe_step=16)
