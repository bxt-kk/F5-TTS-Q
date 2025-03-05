from f5_tts.api import F5TTS
from f5_tts.model import DiT
import torch


f5tts = F5TTS()
f5tts.ema_model.to(torch.float32)
dit_backbone:DiT = f5tts.ema_model.transformer
device = torch.device('cuda:0')
nfe_step = 1024 * 10

times = torch.linspace(0, 1., nfe_step + 1).reshape(-1, 1).to(device)[:-1]
times = times - (torch.cos(torch.pi / 2 * times) - 1 + times)
rows = []
with torch.no_grad():
    for time in times:
        print(time)
        t = dit_backbone.time_embed(time)
        print(t.shape)
        rows.append(t)
X = torch.cat(rows, dim=0)
lr = 0.1

W0 = torch.randn(1024, 512, device=device)
W0 *= 1 / 512**0.5
W1 = torch.randn(512, 1024, device=device)
W1 *= 1 / 512**0.5
W0.requires_grad = True
W1.requires_grad = True

optimizer = torch.optim.SGD([W0, W1], lr=lr)

for step in range(3000):
    _X = X @ W0 @ W1
    loss = torch.nn.functional.mse_loss(X, _X, reduction='mean')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print('loss =', loss.item())

torch.save(dict(
    W0=W0.detach(), W1=W1.detach()), './tests/time_embed_trans.pt')
