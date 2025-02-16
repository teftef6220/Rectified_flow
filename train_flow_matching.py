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


from DiT_GumbelSoftmax.network.dit import DiffusionTransformer
from DiT_GumbelSoftmax.config import Config

# parser
parser = ArgumentParser()
parser.add_argument("--optimizer", type=str, default="RAdamScheduleFree", help="optimizer")
parser.add_argument("--dataset", type=str, default="My_data", help="dataset")
parser.add_argument("--train_use_cfg",type=bool,default=True,help="train with cfg")
parser.add_argument("--dropout_rate",type=float,default=0.1,help="dropout rate")
parser.add_argument("--cfg_scale",type=int,default=2.0,help="cfg scale")
parser.add_argument("--l1_norm",type=bool,default=True,help="l1 norm")
parser.add_argument("--use_dit",type=bool,default=False,help="use diffusion transformer")

args = parser.parse_args()


# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.use_dit:
    config = Config()
    model = DiffusionTransformer(config)

else:
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

# model
    if args.dataset == "Cifar10":
        channel_num = 3
        data_label_num = 10
    elif args.dataset == "MNIST":
        channel_num = 1
        data_label_num = 10
    elif args.dataset == "My_data":
        channel_num = 3
        data_label_num = 26


    unet = Unet(in_channels=channel_num, embedding_channels=64).to(device)
    time_embed = Time_Embed().to(device)
    cond_embed = Cond_Embed(label_num=data_label_num).to(device)
    model = CombinedModel(unet, time_embed, cond_embed).to(device)

# dataloader 
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
epochs = 100
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

total_loss_array,epoch_vec = [],[]

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

            labels = torch.nn.functional.one_hot(labels, data_label_num).float().to(device)

            ## ドロップアウト　if args.train_use_cfg
            if args.train_use_cfg:
                if torch.rand(1).item() < args.dropout_rate: 
                    labels = torch.zeros_like(labels)

            labels_embed  = cond_embed(labels)

            time = torch.rand(1).to(device)
            time_embeds = time_embed(time)

            x_0 = torch.randn_like(images).to(device)
            x_t = time * images + (1 - time) * x_0

            if args.optimizer == "adamW" or args.optimizer == "RAdamScheduleFree":
                v_pred = unet(x_t, time_embeds, labels_embed) 
                loss = criterion(images - x_0 , v_pred)

                if args.l1_norm:
                    l1_lambda = 1e-5  # L1正則化の強度（適宜調整）
                    l1_loss = 0.0
                    for param in unet.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    
                    loss += l1_lambda * l1_loss
                

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

    epoch_vec.append(epoch+1)
    total_loss_array.append(ave_loss/(i+1))

    plt.figure()
    plt.plot(epoch_vec, total_loss_array)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MSE Loss Plot")
    plt.savefig(f"result_{args.dataset}/recon_loss_plot.png")
    plt.close() 


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
        time_embedded = time_embed(torch.linspace(0, 1, 10).unsqueeze(1).to(device))
        cond_embedded = cond_embed(nn.functional.one_hot(torch.arange(data_label_num), data_label_num).float().cuda())

        if args.train_use_cfg:
            uncond_labels = torch.zeros((cond_embedded.size(0), data_label_num), device=device)
            uncond_embedded = cond_embed(uncond_labels)

        for i in range(10):
            
            v_cond = unet(x_0, time_embedded[i], cond_embedded)

            if args.train_use_cfg:
                v_uncond = unet(x_0, time_embedded[i], uncond_embedded)
                v = args.cfg_scale * v_cond - (args.cfg_scale - 1.) * v_uncond
            else:
                v = v_cond
            
            x_0 = x_0 + 0.1 * v
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

        os.makedirs(f"result_{args.dataset}/images", exist_ok=True)
        grid_image.save(f"result_{args.dataset}/images/{epoch}_grid.png")

    if epoch % 20 == 0 and epoch != 0:
        os.makedirs(f"result_{args.dataset}/models", exist_ok=True)
        torch.save(model.state_dict(), f"result_{args.dataset}/models/flow_model_{epoch}_{args.dataset}.pth")


os.makedirs(f"result_{args.dataset}/models", exist_ok=True)
torch.save(model.state_dict(), f"result_{args.dataset}/models/flow_model_last_{args.dataset}.pth")