from f5_tts.api import F5TTS
import torch


f5tts = F5TTS()
print(f5tts.ema_model.to(torch.float32))
print(f5tts.vocoder)
f5tts.infer(
    './src/f5_tts/infer/examples/basic/basic_ref_zh.wav', '对这就是我，万人敬仰的太医真人.',
    gen_text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
    file_wave='tests/infer_try.wav', nfe_step=16)
