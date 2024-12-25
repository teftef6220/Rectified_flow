import torch as th
import torch.nn as nn
import torchvision as tv
import torchvision.transforms.functional as F

from unet import UNet
from embeddings import SinusoidalTimestepEmbedding, ClassEmbedding

# no gradients
with th.no_grad():
    unet = UNet(1)
    time_embed = SinusoidalTimestepEmbedding(256)
    cond_embed = ClassEmbedding(10, 256)
    
    model = nn.ModuleList([unet, time_embed, cond_embed]).cuda()
    model.load_state_dict(th.load("20240317-034318/epoch_41.pt"))
    
    steps = 100
    
    model.eval()
    x_0 = th.randn(10, 1, 28, 28).cuda()
    time_embedded = time_embed(th.linspace(0, 1, steps).cuda())
    cond_embedded = cond_embed(nn.functional.one_hot(th.arange(10), 10).float().cuda())
    
    frames = []
    for i in range(steps):
        frames.append(x_0[8].cpu())
        v = unet(x_0, time_embedded[i], cond_embedded)
        x_0 = x_0 + (1 / steps) * v
    frames.append(x_0[8].cpu())
    
    sample = th.cat([x for x in x_0], dim=2)
    sample = (sample + 1) / 2
    sample = sample.clamp(0, 1) * 255
    sample = sample.to(th.uint8)
    sample = sample.cpu()
    tv.io.write_png(sample, "sample.png")
    
    frames = th.cat(frames, dim=0)
    frames = (frames + 1) / 2
    frames = frames.clamp(0, 1) * 255
    frames = frames.to(th.uint8)
    frames = frames.unsqueeze(1)
    frames = frames.repeat(1, 3, 1, 1)
    frames = F.resize(frames, 256)
    frames = frames.permute(0, 2, 3, 1)
    frames = frames.cpu()
    
    tv.io.write_video("sample.mp4", frames, 30)
    