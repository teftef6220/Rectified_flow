from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import UNet
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from VQVAE.my_data_loader import MyDataLoader
import matplotlib.pyplot as plt

# parser
parser = ArgumentParser()
parser.add_argument("--optimizer", type=str, default="RAdamScheduleFree", help="optimizer")
parser.add_argument("--dataset", type=str, default="My_data", help="dataset")
parser.add_argument("--gen_time_step", type=int, default=100, help="generate time step")
parser.add_argument("--use_cfg",type=bool,default=True,help="train with cfg")
parser.add_argument("--cfg_scale",type=int,default=1.5,help="cfg scale")

args = parser.parse_args()


class Unet(nn.Module):
    def __init__(self, in_channels, embedding_channels=64, time_embed_dim=256, cond_embed_dim=256, depth=4):
        super(Unet, self).__init__()

        self.unet = UNet(
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            cond_embed_dim=cond_embed_dim,
            time_embed_dim=time_embed_dim,
            depth=depth,
            kernel_size=[3,3,3,3,3,3,3],
            layers=[3,3,3,9,3,3,3],
            num_groups=[32] * (depth * 2 - 1) 
        )
    
    def forward(self, x, t, c):
        return self.unet(x, t, c)
    
class Time_Embed(nn.Module):
    def __init__(self, time_embed_dim=256):
        super(Time_Embed, self).__init__()
        self.time_embed = nn.Linear(1,time_embed_dim)
        self.reg = nn.ReLU(inplace=False)
    
    def forward(self, t):
        return self.reg(self.time_embed(t))

class Cond_Embed(nn.Module):
    def __init__(self,label_num=10, cond_embed_dim=256):
        super(Cond_Embed, self).__init__()
        self.cond_embed = nn.Linear(label_num,cond_embed_dim)
        # self.normalize = nn.LayerNorm(cond_embed_dim)
        self.reg = nn.ReLU(inplace=False)
    
    def forward(self, c):
        out = self.cond_embed(c)
        # out = self.normalize(out)
        return self.reg(out)


class CombinedModel(nn.Module):
    def __init__(self, unet, time_embed, cond_embed):
        super(CombinedModel, self).__init__()
        self.unet = unet
        self.time_embed = time_embed
        self.cond_embed = cond_embed

    def forward(self, x, t, c):
        t = self.time_embed(t)
        c = self.cond_embed(c)
        return self.unet(x, t, c)

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model
unet = Unet(in_channels=1, embedding_channels=64).to(device)
time_embed = Time_Embed().to(device)
if args.dataset == "Cifar10":
    unet = Unet(in_channels=3, embedding_channels=64).to(device)
    time_embed = Time_Embed().to(device)
    cond_embed = Cond_Embed(label_num=10).to(device)
elif args.dataset == "MNIST":
    unet = Unet(in_channels=1, embedding_channels=64).to(device)
    time_embed = Time_Embed().to(device)
    cond_embed = Cond_Embed(label_num=10).to(device)
elif args.dataset == "My_data":
    unet = Unet(in_channels=3, embedding_channels=64).to(device)
    time_embed = Time_Embed().to(device)
    cond_embed = Cond_Embed(label_num=26).to(device)

model = CombinedModel(unet, time_embed, cond_embed).to(device)

if args.optimizer == "RAdamScheduleFree":
    from schedulefree import RAdamScheduleFree
    optimizer = RAdamScheduleFree(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

# load_model
model.load_state_dict(torch.load(f"result_{args.dataset}/models/flow_model_last_{args.dataset}_cfg_l1.pth",map_location=device))
with torch.no_grad():
    model.eval()
    if args.optimizer == "RAdamScheduleFree":
        optimizer.eval()
    if args.dataset == "Cifar10":
        x_0 = torch.randn(10, 3, 32, 32).to(device)
    elif args.dataset == "MNIST":
        x_0 = torch.randn(10, 1, 28, 28).to(device)
    elif args.dataset == "My_data":
        x_0 = torch.randn(26, 3, 64, 64).to(device)
    
    time_embedded = time_embed(torch.linspace(0, 1, args.gen_time_step).unsqueeze(1).to(device))
    cond_embedded = cond_embed(nn.functional.one_hot(torch.arange(26), 26).float().cuda())
    
    if args.use_cfg:
        uncond_labels = torch.zeros((cond_embedded.size(0), 26), device=device)
        uncond_embedded = cond_embed(uncond_labels)

    for i in tqdm(range(args.gen_time_step)):
        v_cond = unet(x_0, time_embedded[i], cond_embedded)

        if args.use_cfg:
            v_uncond = unet(x_0, time_embedded[i], uncond_embedded)
            v = args.cfg_scale * v_cond - (args.cfg_scale - 1.) * v_uncond
        else:
            v = v_cond

        x_0 = x_0 + (1/args.gen_time_step) * v
        # x_0 = x_0.clamp(-1, 1)
    sample = (x_0 + 1) / 2
    sample.clamp_(0, 1)

    pil_images = [transforms.functional.to_pil_image(x) for x in (sample * 255).to(torch.uint8)]
    #Save image in one image
    cols , rows = 5, 5
    img_width, img_height = pil_images[0].size
    grid_width = img_width * cols 
    grid_height = img_height * rows
    grid_image = Image.new("RGB", (grid_width, grid_height))

    for idx, img in enumerate(pil_images):
        x_offset = (idx % cols) * img_width  
        y_offset = (idx // cols) * img_height  
        grid_image.paste(img, (x_offset, y_offset))

    os.makedirs(f"result_{args.dataset}/test_images", exist_ok=True)
    grid_image.save(f"result_{args.dataset}/test_images/test_grid.png")

