import torch
import torch.nn as nn
import torch.nn.functional as F
from .model.modules import DiTBlock


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


class LinearSiLU(nn.Module):

    def __init__(
            self,
            in_dim:  int,
            out_dim: int,
        ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
        )

    def forward(
            self,
            x:torch.Tensor,
            t=None,
            mask=None,
            rope=None,
        ) -> torch.Tensor:
        return self.net(x)


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
            LayerNormChannel(in_channels))

    def forward(
            self,
            x:torch.Tensor,
            t=None,
            mask=None,
            rope=None,
        ) -> torch.Tensor:
        # x: b, n, d
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x0 = x.transpose(1, 2) # b, d, n
        x = self.normal_active(self.conv_proj(x0))
        for linear_conv in self.linear_convs:
            x = x + linear_conv(x)
        x = self.normal_active(x)
        x = x0 + self.conv_ff_norm(x)
        x =  x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        return x


def quant_transformer_blocks(dit_blocks:nn.ModuleList, status:list[int]) -> tuple[nn.ModuleList, list[int]]:
    status_q = []
    blocks = []
    for si, st in enumerate(status):
        if st == 2:
            if si > 0 and status[si - 1] != 2:
                blocks.append(LinearSiLU(512, 1024))
                status_q.append(st)
            blocks.append(dit_blocks[si])
            status_q.append(st)
        elif st == 1:
            if si > 0 and status[si - 1] == 2:
                blocks.append(LinearSiLU(1024, 512))
                status_q.append(st)
            blocks.append(DiTBlock(
                dim=512, heads=16, dim_head=32, ff_mult=2, dropout=0.1))
            status_q.append(st)
        elif st == 0:
            if si > 0 and status[si - 1] == 1:
                blocks.append(LinearConvsBlock(
                    512, 512, num_layers=3, groups=16))
                status_q.append(st)
    return nn.ModuleList(blocks), status_q
