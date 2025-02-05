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
from vae import VAE_Encoder, VAE_Decoder, VAE_Decoder_UV,VAE


# load_model

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

# only use Decoder
if __name__ == '__main__':
    input_dim = 28 * 28 * 1
    hidden_dim = 200
    latent_dim = 26
    batch_size = 64
    epochs = 20
    lr = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_every_n_epochs = 1

    model = VAE(in_dim = input_dim,h_dim=hidden_dim,out_dim=latent_dim,is_train=False).to(device)
    model = load_model(model, './VAE/model/VAE_10.pth').to(device)

    model.eval()
    with torch.no_grad():
        x = torch.randn(64, latent_dim).to(device)
        degrees = (60,60)  # Rotation range
        translate = (1.0, 1.0)  # Translation as a fraction of image size
        scale = (1.0, 1.0)  # Scaling range
        out = model.decoder(x, (64, 1, 28, 28),degrees = degrees,translate = translate,scale = scale)
        out = out.view(-1,1,28,28)
        grid_im = torchvision.utils.make_grid(out)
        # save image
        torchvision.utils.save_image(grid_im,f"./VAE/result/sample_images.png")
        
    