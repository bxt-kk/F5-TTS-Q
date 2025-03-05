from f5_tts.api import F5TTS
import torch


f5tts = F5TTS()

f5tts.ema_model.to(torch.float32)
# print(f5tts.vocoder)
torch.cuda.synchronize(f5tts.device)
print(f5tts.ema_model.transformer.transformer_blocks[0])
