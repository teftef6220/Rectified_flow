import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from vae_utils import random_affine_transform


class VAE_Encoder(nn.Module):
    def __init__(self, in_dim,h_dim=400,out_dim=26):
        super().__init__()

        self.linear1 = nn.Linear(in_dim,h_dim)
        self.relu = nn.ReLU()
        self.linear_mu = nn.Linear(h_dim,out_dim)
        self.linear_var = nn.Linear(h_dim,out_dim)


    def forward(self,x):
        h = self.relu(self.linear1(x))
        mu = self.linear_mu(h)
        log_var = self.linear_var(h)

        sigma = torch.exp(log_var * 0.5)

        return mu,sigma

class VAE_Decoder(nn.Module):
    def __init__(self,in_dim,h_dim=200,out_dim=400):
        super().__init__()
        self.linear1 = nn.Linear(in_dim,h_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(h_dim,out_dim)
        self.relu_2 = nn.ReLU()
        self.linear3 = nn.Linear(out_dim,28*28*1)
        self.normalize = nn.Sigmoid()

    def forward(self,z):
        h = self.relu(self.linear1(z))
        h = self.relu_2(self.linear2(h))
        out = self.normalize(self.linear3(h))
        return out


class VAE_Decoder_UV(nn.Module):
    def __init__(self,in_dim = 28*28*28,h_dim= 28*28 ,out_dim=400,degrees=(-90,90),translate=(1.0,1.0),scale=(1.0,1.0),is_train=True):
        super().__init__()
        self.linear1 = nn.Linear(in_dim,h_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(h_dim,out_dim)
        self.relu_2 = nn.ReLU()
        self.linear3 = nn.Linear(out_dim,28*28*1)
        self.normalize = nn.Sigmoid()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.is_train = is_train

    def forward(self,z,img_shape,degrees=None,translate=None,scale=None):

        B,_,H,W = img_shape

        u = torch.linspace(-1.0 , 1.0 ,steps = H, device=z.device).view(1, 1, H, 1)
        v = torch.linspace(-1.0 , 1.0 ,steps = W, device=z.device).view(1, 1, 1, W)
        uv = torch.cat(torch.broadcast_tensors(u, v), dim=1) # matrix shape

        # Affine transformation uv 
        # degrees = (-30, 30)  # Rotation range
        # translate = (0.5, 0.5)  # Translation as a fraction of image size
        # scale = (0.8, 1.2)  # Scaling range
        if self.is_train:
            uv = random_affine_transform(z,uv, self.degrees, self.translate, self.scale, self.is_train)
        else:
            uv = random_affine_transform(z,uv, degrees, translate, scale, self.is_train)
        

        # z = z.view(B, -1, 1, 1)
        # z = z.expand(-1, -1, H, W)

        # uvz = torch.cat([uv.expand(B, -1, -1, -1), z], dim=1)

        # z_reduced = z[:, :-2, :, :] 
        # uv_expanded = uv.expand(B, -1, -1, -1)
        # uvz = torch.cat([z_reduced, uv_expanded], dim=1)
        # flatten
        # uvz = uvz.permute(0, 2, 3, 1).reshape(uvz.shape[0],-1)


        h = self.relu(self.linear1(uv.view(B,-1)))
        h = self.relu_2(self.linear2(h))
        out = self.normalize(self.linear3(h))

        # y = y.view(batch_size, H, W, 1).permute(0, 3, 1, 2)
        return out


def reparameterize(mu,sigma):
    eps = torch.randn_like(sigma)
    return mu + sigma * eps

class VAE(nn.Module):
    def __init__(self,in_dim,h_dim=400,out_dim=26,is_train=True):
        super().__init__()

        self.is_train = is_train

        self.encoder = VAE_Encoder(in_dim,h_dim,out_dim)
        # self.decoder = VAE_Decoder(out_dim,200,400)
        self.decoder = VAE_Decoder_UV(28*28*28,28*28,400,is_train = self.is_train)



    def vae(self,x):
        mu,sigma = self.encoder(x)
        z = reparameterize(mu,sigma)
        out = self.decoder(z,x.view(-1,1,28,28).shape)
        # out = self.decoder(z,x.shape)

        return out,mu,sigma
    

if __name__ == '__main__':
    input_dim = 28 * 28 * 1
    hidden_dim = 200
    latent_dim = 26
    batch_size = 64
    epochs = 30
    lr = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_every_n_epochs = 10

    # Load dateset Mnist
    transform = transforms.Compose([
        # transforms.Pad(padding=2, fill=0),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,))
    ])

    data = datasets.MNIST('../Mnist_data',train=True,download=True,transform=transform)

    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )

    import torch.nn.init as init
    # He 初期化
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #         init.zeros_(m.bias)

    model = VAE(in_dim = input_dim,h_dim=hidden_dim,out_dim=latent_dim,is_train=True).to(device)
    # model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(),lr=lr)

    # train
    for epoch in range(epochs):
        model.train()

        total_loss = []
        L_MSE_Total = []
        L_KLD_Total = []

        for x,_ in tqdm(train_loader):
            x = x.view(-1,input_dim).to(device)


            optimizer.zero_grad()
            out,mu,sigma = model.vae(x)
            L_MSE = F.mse_loss(out, x, reduction='sum')
            L_KLD = - torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)

            loss = (L_MSE + L_KLD) / x.size(0)
            # print(L_MSE , L_KLD,"-------loss",loss)

            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            L_MSE_Total.append(L_MSE.item())
            L_KLD_Total.append(L_KLD.item())

            # print(sum(L_MSE_Total) ,sum(L_KLD_Total))

        ave_loss = sum(total_loss) / len(total_loss)
        ave_L_MSE = sum(L_MSE_Total) / len(total_loss)
        ave_L_KLD = sum(L_KLD_Total) / len(total_loss)
        print(f"epoch : {epoch+1}, loss : {ave_loss:.3f}")
        print(f"epoch : {epoch+1}, L_MSE : {ave_L_MSE:.3f}")
        print(f"epoch : {epoch+1}, L_KLD : {ave_L_KLD:.3f}")

        if (epoch+1) % save_every_n_epochs == 0:
            print(f"save model at epoch : {epoch+1}")
            torch.save(model.state_dict(),f"./VAE/model/VAE_{epoch+1}.pth")


        # save image
        model.eval()
        with torch.no_grad():
            sample_size = 64
            z = torch.randn(sample_size,latent_dim).to(device)
            out = model.decoder(z,(sample_size,1,28,28))
            out = out.view(-1,1,28,28)

            grid_im = torchvision.utils.make_grid(out)
            # save image
            torchvision.utils.save_image(grid_im,f"./VAE/result/{epoch+1}.png")