from f5_tts.api import F5TTS
import torch


f5tts = F5TTS()

f5tts.ema_model.to(torch.float32)
# print(f5tts.vocoder)
torch.cuda.synchronize(f5tts.device)
dit = f5tts.ema_model.transformer
# dit.init_buffer_deque(2000)
# exit()
f5tts.infer(
    './src/f5_tts/infer/examples/basic/basic_ref_zh.wav', '对，这就是我，万人敬仰的太乙真人.',
    gen_text="I don't really care what you call me.",
    file_wave='tests/infer_try.wav', nfe_step=16)
print(dit.buffer_deque)
