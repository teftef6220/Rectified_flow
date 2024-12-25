import torch as th
import torch.nn as nn
import torch.nn.functional as F

class LearnableTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(1, embed_dim)
        self.silu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.max_period = 10000
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.silu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        half = self.embed_dim // 2
        freqs = th.exp(th.arange(start=0, end=half).cuda().float() / half) / th.tensor(self.max_period).cuda()
        args = x[:, None] * freqs[None]
        x = th.cat([th.cos(args), th.sin(args)], dim=-1).squeeze(1)
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(num_classes, embed_dim)
        self.silu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x