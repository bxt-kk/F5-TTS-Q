import torch
import torch.nn as nn
import torch.nn.functional as F
from f5_tts.api import F5TTS
from f5_tts.model import DiT
from f5_tts.model.modules import DiTBlock
from x_transformers.x_transformers import RotaryEmbedding


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
            LayerNormChannel(in_channels),
            nn.SiLU())

    def forward(
            self,
            x:torch.Tensor,
            t=None,
            mask=None,
            rope=None,
        ) -> torch.Tensor:
        # x: b, n, d
        x0 = x.transpose(1, 2) # b, d, n
        x = self.normal_active(self.conv_proj(x0))
        for linear_conv in self.linear_convs:
            x = x + linear_conv(x)
        x = self.normal_active(x)
        x = x0 + self.conv_ff_norm(x)
        return x.transpose(1, 2)


def quant_transformer_blocks(dit:DiT, status:list[int]) -> tuple[nn.ModuleList, list[int]]:
    status_q = []
    blocks = []
    for si, st in enumerate(status):
        if st == 2:
            if si > 0 and status[si - 1] != 2:
                blocks.append(LinearSiLU(512, 1024))
                status_q.append(st)
            blocks.append(dit.transformer_blocks[si])
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


if __name__ == "__main__":
    device = torch.device('cuda:0')
    status = [2, 2, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]

    f5tts = F5TTS()

    f5tts.ema_model.to(torch.float32)
    dit:DiT = f5tts.ema_model.transformer

    blocks, status_q = quant_transformer_blocks(dit, status)
    blocks.to(device)
    rotary_embed_q = RotaryEmbedding(32).to(device)

    t = torch.randn(1, 1024).to(device)
    t_q = torch.randn(1, 512).to(device)

    import time
    x = torch.randn(1, 995, 1024).to(device)
    with torch.no_grad():
        rope = dit.rotary_embed.forward_from_seq_len(995)
        for block in dit.transformer_blocks:
            x = block(x, t, rope=rope)
        x.cpu()
        clock = time.time()
        for block in dit.transformer_blocks:
            x = block(x, t, rope=rope)
        x.cpu()
        print('src blocks delta:', time.time() - clock)
    x = torch.randn(1, 995, 1024).to(device)
    with torch.no_grad():
        rope = dit.rotary_embed.forward_from_seq_len(995)
        rope_q = rotary_embed_q.forward_from_seq_len(995)
        for bid, block in enumerate(blocks):
            _t = t
            if status_q[bid] == 1:
                x = block(x, t_q, rope=rope_q)
            else:
                x = block(x, t, rope=rope)
        x.cpu()
        # print(rope[0].shape, rope[1], rope_q[0].shape, rope_q[1])
        clock = time.time()
        for bid, block in enumerate(blocks):
            _t = t
            if status_q[bid] == 1:
                x = block(x, t_q, rope=rope_q)
                # print(x.shape, t_q.shape, type(block), status_q[bid], bid)
            else:
                x = block(x, t, rope=rope)
                # print(x.shape, t.shape, type(block), status_q[bid], bid)
        x.cpu()
        print('quant blocks delta:', time.time() - clock)

    # head_dim = 64
    # num_heads = 16
    # in_dim = 1024
    # seq_len = 1000
    # t_test = torch.randn(1, in_dim).to(device)
    # inputs = torch.randn(1, seq_len, in_dim).to(device)
    #
    # block = LinearConvsBlock(in_dim, head_dim * num_heads, num_layers=4, groups=num_heads).to(device)
    # import time
    #
    # with torch.no_grad():
    #     outputs = block(inputs)
    # clock = time.time()
    # with torch.no_grad():
    #     outputs = block(inputs)
    # outputs.cpu()
    # print('debug[linear-conv]:', time.time() - clock)
    # print(outputs.shape)
    #
    # ditblock = DiTBlock(dim=1024, heads=16, dim_head=64, ff_mult=2, dropout=0.1).to(device).eval()
    # with torch.no_grad():
    #     outputs = ditblock(inputs, t=t_test)
    # clock = time.time()
    # with torch.no_grad():
    #     outputs = ditblock(inputs, t=t_test)
    # outputs.cpu()
    # print('debug[dit-block]:', time.time() - clock)
    # print(outputs.shape)
    #
    # in_dim = 512
    # head_dim = 32
    # num_heads = 16
    # t_test = torch.randn(1, in_dim).to(device)
    # inputs = torch.randn(1, seq_len, in_dim).to(device)
    #
    # ditblock_q = DiTBlock(dim=in_dim, heads=16, dim_head=32, ff_mult=2, dropout=0.1).to(device).eval()
    # with torch.no_grad():
    #     outputs = ditblock_q(inputs, t=t_test)
    # clock = time.time()
    # with torch.no_grad():
    #     outputs = ditblock_q(inputs, t=t_test)
    # outputs.cpu()
    # print('debug[dit-block Q]:', time.time() - clock)
    # print(outputs.shape)
    # # print(ditblock)
    # # print(ditblock_q)
    #
    # block_q = LinearConvsBlock(in_dim, head_dim * num_heads, num_layers=4, groups=num_heads).to(device)
    #
    # with torch.no_grad():
    #     outputs = block_q(inputs)
    # clock = time.time()
    # with torch.no_grad():
    #     outputs = block_q(inputs)
    # outputs.cpu()
    # print('debug[linear-conv Q]:', time.time() - clock)
    # print(outputs.shape)
