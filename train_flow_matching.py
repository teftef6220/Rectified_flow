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

# parser
parser = ArgumentParser()
parser.add_argument("--optimizer", type=str, default="adamW", help="optimizer")
parser.add_argument("--dataset", type=str, default="My_data", help="dataset")

args = parser.parse_args()




class Unet(nn.Module):
    def __init__(self, in_channels, embedding_channels=64, time_embed_dim=256, cond_embed_dim=256, depth=3):
        super(Unet, self).__init__()

        self.unet = UNet(
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            cond_embed_dim=cond_embed_dim,
            time_embed_dim=time_embed_dim,
            depth=depth,
            kernel_size=[3,3,3,3,3],
            layers=[3,3,9,3,3],
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

# dataloader CIFAR10
if args.dataset == "Cifar10":
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)


# dataloader MNIST
elif args.dataset == "MNIST":
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)


elif args.dataset == "My_data":
    batch_size = 8
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = MyDataLoader(data_dir='./make_dataset/stamps', transform=transform )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

os.makedirs(f"result_{args.dataset}", exist_ok=True)

print("Training On ", device)
# train
epochs = 30
criterion = torch.nn.MSELoss()

if args.optimizer == "adamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
elif args.optimizer == "RAdamScheduleFree":
    from schedulefree import RAdamScheduleFree
    optimizer = RAdamScheduleFree(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
elif args.optimizer == "LBFGS":
    def closure(x,y):
        optimizer.zero_grad()               # init grad
        output = model(x)                   # Model
        loss = criterion(output, y)         # Loss
        loss.backward()                     # Grad
        return loss
    
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    if args.optimizer == "RAdamScheduleFree":
        optimizer.train()
    total_loss = 0.0
    #tqdm
    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            labels = labels.to(device)
            images = images.to(device) #(batch, 1, 28, 28)

            labels = torch.nn.functional.one_hot(labels, 26).float().to(device)
            labels_embed  = cond_embed(labels)

            time = torch.rand(1).to(device)
            time_embeds = time_embed(time)

            x_0 = torch.randn_like(images).to(device)
            x_t = time * images + (1 - time) * x_0

            if args.optimizer == "adamW" or args.optimizer == "RAdamScheduleFree":
                v_pred = unet(x_t, time_embeds, labels_embed) 
                loss = criterion(images - x_0 , v_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif args.optimizer == "LBFGS":
                def closure():
                    optimizer.zero_grad()  # init grad
                    v_pred = unet(x_t, time_embeds, labels_embed)  # model
                    loss = criterion(images - x_0, v_pred)  # loss
                    loss.backward(retain_graph=True)  # Grad
                    return loss
                # L-BFGSの更新
                loss = optimizer.step(closure)

            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    ave_loss = total_loss / len(train_loader) * batch_size
    print(f"Epoch: {epoch+1}, Loss: {ave_loss/(i+1)}")

    # check inference
    
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
        time_embedded = time_embed(torch.linspace(0, 1, 26).unsqueeze(1).to(device))
        cond_embedded = cond_embed(nn.functional.one_hot(torch.arange(26), 26).float().cuda())
        for i in range(26):
            v = unet(x_0, time_embedded[i], cond_embedded)
            x_0 = x_0 + 0.1 * v
        sample = (x_0 + 1) / 2
        sample.clamp_(0, 1)

        pil_images = [transforms.functional.to_pil_image(x) for x in (sample * 255).to(torch.uint8)]
        #Save image in one image
        cols , rows = 5, 2
        img_width, img_height = pil_images[0].size
        grid_width = img_width * cols 
        grid_height = img_height * rows
        grid_image = Image.new("RGB", (grid_width, grid_height))

        for idx, img in enumerate(pil_images):
            x_offset = (idx % cols) * img_width  
            y_offset = (idx // cols) * img_height  
            grid_image.paste(img, (x_offset, y_offset))

        os.makedirs(f"result_{args.dataset}/images", exist_ok=True)
        grid_image.save(f"result_{args.dataset}/images/{epoch}_grid.png")

    if epoch % 10 == 0 and epoch != 0:
        os.makedirs(f"result_{args.dataset}/models", exist_ok=True)
        torch.save(model.state_dict(), f"result_{args.dataset}/models/flow_model_{epoch}_{args.dataset}.pth")


os.makedirs(f"result_{args.dataset}/models", exist_ok=True)
torch.save(model.state_dict(), f"result_{args.dataset}/models/flow_model_last_{args.dataset}.pth")