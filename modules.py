import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormChannel(nn.Module):
    def __init__(
            self,
            dim: int,
            eps: float = 1e-5,
        ):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.bias   = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.shape  = (dim, )
        self.eps    = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.transpose(1, 2),
            normalized_shape=self.shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps).transpose(1, 2)


class LinearConvsBlock(nn.Module):

    def __init__(self,
            in_channels:     int,
            hidden_channels: int,
            num_layers:      int,
            groups:          int,
        ):

        super().__init__()

        self.conv_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        blocks = []
        for l in range(num_layers):
            dilation = 2**l
            blocks.append(nn.Sequential(
                nn.ZeroPad1d((0, (2 + 2 * (dilation - 1) - 1) // 2 + 1)),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=2, dilation=dilation, groups=groups),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            ))
        self.linear_convs = nn.ModuleList(blocks)
        self.normal_active = nn.Sequential(
            LayerNormChannel(hidden_channels),
            nn.SiLU())
        self.conv_ff_norm = nn.Sequential(
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
            LayerNormChannel(in_channels),
            nn.SiLU())


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: b, n, d
        x0 = x.transpose(1, 2) # b, d, n
        x = self.normal_active(self.conv_proj(x0))
        for linear_conv in self.linear_convs:
            x = x + linear_conv(x)
        x = self.normal_active(x)
        x = x0 + self.conv_ff_norm(x)
        return x.transpose(1, 2)


device = torch.device('cuda:0')

head_dim = 64
num_heads = 16
in_dim = 1024
seq_len = 1000
t_test = torch.randn(1, in_dim).to(device)
inputs = torch.randn(1, seq_len, in_dim).to(device)

block = LinearConvsBlock(in_dim, head_dim * num_heads, num_layers=4, groups=num_heads).to(device)
import time

with torch.no_grad():
    outputs = block(inputs)
clock = time.time()
with torch.no_grad():
    outputs = block(inputs)
outputs.cpu()
print('debug[linear-conv]:', time.time() - clock)
print(outputs.shape)

from f5_tts.model.modules import DiTBlock

ditblock = DiTBlock(dim=1024, heads=16, dim_head=64, ff_mult=2, dropout=0.1).to(device).eval()
with torch.no_grad():
    outputs = ditblock(inputs, t=t_test)
clock = time.time()
with torch.no_grad():
    outputs = ditblock(inputs, t=t_test)
outputs.cpu()
print('debug[dit-block]:', time.time() - clock)
print(outputs.shape)

in_dim = 512
head_dim = 32
num_heads = 16
t_test = torch.randn(1, in_dim).to(device)
inputs = torch.randn(1, seq_len, in_dim).to(device)

ditblock_q = DiTBlock(dim=in_dim, heads=16, dim_head=32, ff_mult=2, dropout=0.1).to(device).eval()
with torch.no_grad():
    outputs = ditblock_q(inputs, t=t_test)
clock = time.time()
with torch.no_grad():
    outputs = ditblock_q(inputs, t=t_test)
outputs.cpu()
print('debug[dit-block Q]:', time.time() - clock)
print(outputs.shape)
# print(ditblock)
# print(ditblock_q)

block_q = LinearConvsBlock(in_dim, head_dim * num_heads, num_layers=4, groups=num_heads).to(device)

with torch.no_grad():
    outputs = block_q(inputs)
clock = time.time()
with torch.no_grad():
    outputs = block_q(inputs)
outputs.cpu()
print('debug[linear-conv Q]:', time.time() - clock)
print(outputs.shape)
