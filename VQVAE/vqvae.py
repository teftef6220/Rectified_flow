import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import os
from torch.utils.data import DataLoader
from vqvae_utils import ViTEncoder , TransformerDecoder

class Residual(nn.Module):
    def __init__(self,in_channels,hidden_channels,residual_hiddens_num):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,residual_hiddens_num,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(residual_hiddens_num,hidden_channels,kernel_size=3,stride=1,padding=1)
        self.ReLU_1 = nn.ReLU()
        self.ReLU_2 = nn.ReLU()


    def forward(self,x):
        h = self.conv1(x)
        h = self.ReLU_1(h)
        h = self.conv2(h)
        # h = self.ReLU_2(h)

        return x + h

class Residual_Block(nn.Module):
    def __init__(self,in_channels,hidden_channels,residual_hidden_num,residual_hidden_dim):
        super().__init__()
        self.residual_hidden_num = residual_hidden_num
        self.residuals = nn.ModuleList([Residual(in_channels,hidden_channels,residual_hidden_dim) for _ in range(residual_hidden_num)])
        self.ReLU = nn.ReLU()

    def forward(self,x):
        for i in range(self.residual_hidden_num):
            x = self.residuals[i](x)
        return self.ReLU(x)


class VQVAE_Encoder(nn.Module):
    def __init__(self,in_channels,hidden_channels,residual_hiddens_num,residual_hidden_dim,name = None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.residual_hiddens_num = residual_hiddens_num
        self.residual_hidden_dim = residual_hidden_dim

        self.enc_1 = nn.Conv2d(in_channels,hidden_channels // 2,kernel_size=4,stride=2,padding=1)
        self.enc_2 = nn.Conv2d(hidden_channels // 2,hidden_channels,kernel_size=4,stride=2,padding=1)

        # self.enc_1 = nn.Conv2d(in_channels,hidden_channels,kernel_size=5,stride=4,padding=1)
        self.ReLU1 = nn.ReLU()
        self.enc_3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.residual = Residual_Block(hidden_channels,hidden_channels,residual_hiddens_num,residual_hidden_dim)
        self.ReLU2 = nn.ReLU()


    def forward(self,x):# 64 * 3 * 32* 32
        h = self.enc_1(x) # 64 * 64 * 16 * 16
        h = self.ReLU1(h)
        h = self.enc_2(h) # 64 * 128 * 8 * 8
        h = self.ReLU2(h)
        h = self.enc_3(h)
        h = self.residual(h)
        return h # 64 * 128 * 8 * 8


class VQVAE_Decoder(nn.Module):
    def __init__(self,in_channels,hidden_channels,residual_hiddens_num,residual_hidden_dim,name = None):
        super().__init__()

        self.in_channles = in_channels
        self.hidden_channels = hidden_channels
        self.residual_hiddens_num = residual_hiddens_num
        self.residual_hidden_dim = residual_hidden_dim

        self.dec_1 = nn.Conv2d(in_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
        self.residual = Residual_Block(hidden_channels,hidden_channels,residual_hiddens_num,residual_hidden_dim)

        self.dec_2 = nn.ConvTranspose2d(hidden_channels,hidden_channels // 2,kernel_size=4,stride=2,padding=1)
        self.dec_3 = nn.ConvTranspose2d(hidden_channels // 2,3,kernel_size=4,stride=2,padding=1)
        self.ReLU = nn.ReLU()


    def forward(self,x):  # 64 * 32 * 8 * 8
        h = self.dec_1(x)
        h = self.residual(h)
        h = self.dec_2(h)
        h = self.ReLU(h)
        h = self.dec_3(h)
        h = torch.sigmoid(h)
        return h # 64 * 3 * 32* 32
    

    


# 統合するよ
class VQVAE_all(nn.Module):
    def __init__(self,encoder,decoder,hidden_dim,embedding_dim,num_embeddings,commitment_cost,data_variance):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim  
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.data_variance = data_variance
        self.embedding = nn.Embedding(num_embeddings,embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings,1/num_embeddings)

        self.pre_vq_conv1 = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1, stride=1)

    def quantize(self,x):
        input_shape = x.shape

        inputs = x.permute(0,2,3,1).contiguous()  # [B,C,H,W] -> [B,H,W,C] 
        inputs_shape = inputs.size()

        flat_input = inputs.view(-1,self.embedding_dim) # [B*H*W,C] [4096,32]

        distance = (torch.sum(flat_input**2,dim=1,keepdim=True)
                    +torch.sum(self.embedding.weight**2,dim=1) # self.embedding.weight = [128,32]
                    -2 * torch.matmul(flat_input,self.embedding.weight.t())) # ||x-y|^2  = ||x||^2 + ||y||^2 - 2 x y
        
        
        # flat_input_norm = F.normalize(flat_input, p=2, dim=1)         # shape: (B*H*W, C)
        # embedding_norm  = F.normalize(self.embedding.weight, p=2, dim=1)  # shape: (num_embeddings, C)
        # cos_sim = torch.matmul(flat_input_norm, embedding_norm.t())
        # distance = 1 - cos_sim


        indices = torch.argmin(distance,dim=1).unsqueeze(1)
        encodings = torch.zeros(indices.shape[0],self.num_embeddings,device=x.device)
        encodings.scatter_(1,indices,1)
        quantized = torch.matmul(encodings,self.embedding.weight).view(inputs_shape)


        e_latent_loss = F.mse_loss(quantized.detach(),inputs)
        q_latent_loss = F.mse_loss(quantized,inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        avg_probs = torch.mean(encodings,dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return {'distances': distance,
                'quantize': quantized,
                'loss': loss,
                'encodings': encodings,
                'indices': indices,
                'perplexity': perplexity
                }
    
    def decode(self,z):
        return self.decoder(z)
    
    def forward(self,inputs):
        x = self.encoder(inputs)
        z = self.pre_vq_conv1(x)
        vq_output = self.quantize(z)

        x_reconstructed = self.decoder(vq_output['quantize'])
        reconstructed_error = torch.mean(torch.square(x_reconstructed - inputs) / self.data_variance)

        loss = reconstructed_error + vq_output['loss']
        return {
            'z': z,
            'x_reconstructed': x_reconstructed,
            'loss': loss,
            'reconstructed_error': reconstructed_error,
            'VQ_error': vq_output['loss'],
            'vq_output': vq_output
        }




if __name__ == "__main__": 
    batch_size = 4
    image_size = 80

    hidden_dim = 128
    residual_hidden_dim = 32
    num_residual_lyers = 2

    embedding_dim = 32
    num_embedding = 128 #K
    commitence_cost = 0.25
    use_ema = False
    ema_decay = 0.95
    lr = 1e-4
    epochs = 50

    output_dir = "./vqvae_generated_images"
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device : ",device)

    # Load dateset CIFAR10
    # transform = transforms.Compose([
    #     # transforms.Pad(padding=2, fill=0),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5,),(0.5,))
    # ])

    #prsk_data
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # data = datasets.CIFAR10('.data/CIFAR10_data',train=True,download=True,transform=transform)

    # CiFar10
    # train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

    # prsk_data
    from my_data_loader import MyDataLoader
    dataset = MyDataLoader(data_dir='./make_dataset/stamps', transform=transform )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model

    encoder = VQVAE_Encoder(in_channels = 3, hidden_channels = hidden_dim, residual_hiddens_num = num_residual_lyers, residual_hidden_dim = residual_hidden_dim)
    decoder = VQVAE_Decoder(in_channels = embedding_dim, hidden_channels = hidden_dim, residual_hiddens_num = num_residual_lyers, residual_hidden_dim = residual_hidden_dim)


    # encoder = ViTEncoder(
    #     in_channels=3,
    #     patch_size=4,
    #     embed_dim=128,
    #     image_size=32,
    #     n_layers=2,      # 簡易化のため少なめ
    #     n_heads=4,
    #     dim_feedforward=256,
    #     dropout=0.1
    # )
    # decoder = TransformerDecoder(
    #     out_channels=3,
    #     patch_size=4,
    #     embed_dim=128,
    #     image_size=32,
    #     n_layers=2,
    #     n_heads=4,
    #     dim_feedforward=256,
    #     dropout=0.1
    # )

    model = VQVAE_all(encoder = encoder, decoder = decoder, hidden_dim = hidden_dim, embedding_dim = embedding_dim,num_embeddings = num_embedding ,commitment_cost =commitence_cost, data_variance = 1.0).to(device)


    optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))


    total_loss_Recon , total_loss_VQ ,epoch_vec= [],[],[]
    # train
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        reconstruction_err = []
        perplexity_err =[]

        for x,_ in tqdm(train_loader):
            x = x.to(device)
            out = model(x)


            optimizer.zero_grad()
            loss = out['loss']

            loss.backward()
            optimizer.step()

        print("loss : ",loss.item())
        print("Recon Loss : ",out['reconstructed_error'].item()," , VQ_Loss : ",out['VQ_error'].item())
        total_loss_Recon.append(out['reconstructed_error'].item())
        total_loss_VQ.append(out['VQ_error'].item())
        epoch_vec.append(epoch)

        # Recon Lossのプロット
        plt.figure()
        plt.plot(epoch_vec, total_loss_Recon)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Recon Loss Plot")
        plt.savefig(f"{output_dir}/recon_loss_plot.png")
        plt.close()  # グラフを閉じてリセット

        # VQ Lossのプロット
        plt.figure()
        plt.plot(epoch_vec, total_loss_VQ)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VQ Loss Plot")
        plt.savefig(f"{output_dir}/vq_loss_plot.png")
        plt.close()  # グラフを閉じてリセット


        print("input_image")
        grid_im_x = torchvision.utils.make_grid(x)
        save_image(grid_im_x, f"{output_dir}/Orig_epoch_{(epoch+1):03}.png")

        print("Recon Image :")
        grid_im_recon = torchvision.utils.make_grid(out['x_reconstructed'])
        save_image(grid_im_recon, f"{output_dir}/Recon_epoch_{(epoch+1):03}.png")

        # model.eval()
        # with torch.no_grad():
        #     random_indices = torch.randint(
        #         0, 128, size=(64, 8, 8), device=device
        #     )
        #     one_hot = F.one_hot(random_indices, num_classes=128).float()
        #     flat_one_hot = one_hot.view(-1, 128)  # [64*8*8, num_embeddings]
        #     random_quantized = torch.matmul(flat_one_hot, quantizer.embedding.weight)
        #     random_quantized = random_quantized.view(64, 8, 8, embedding_dim)
        #     random_quantized = random_quantized.permute(0, 3, 1, 2).contiguous()
        #     generated_images = model.decoder(random_quantized).cpu()

        #     grid_im = torchvision.utils.make_grid(generated_images)
        #     # plt.imshow(grid_im.permute(1,2,0))
        #     # plt.show()

        #     os.makedirs(output_dir, exist_ok=True)
        #     save_image(grid_im, f"{output_dir}/epoch_{epoch+1}_generated.png")

        # print(f"Epoch {epoch+1}: Generated images saved.")





    torch.save(model.state_dict(), "vqvae_model.pth")









# class VectorQuantizer(nn.Module):
#     def __init__(self,num_embeddings,embedding_dim,commitment_cost):
#         super().__init__()

#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = commitment_cost
        

#         self.embedding = nn.Embedding(num_embeddings,embedding_dim)
#         self.embedding.weight.data.uniform_(-1/num_embeddings,1/num_embeddings)

#     def forward(self,x):
#         input_shape = x.shape

#         inputs = x.permute(0,2,3,1).contiguous()
#         inputs_shape = inputs.size()

#         flat_input = inputs.view(-1,self.embedding_dim)

#         distance = (torch.sum(flat_input**2,dim=1,keepdim=True)
#                     +torch.sum(self.embedding.weight**2,dim=1)
#                     -2 * torch.matmul(flat_input,self.embedding.weight.t())) # ||x-y|^2  = ||x||^2 + ||y||^2 - 2 x y
#         encoding_indices = torch.argmin(distance,dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0],self.num_embeddings,device=x.device)
#         encodings.scatter_(1,encoding_indices,1)
#         quantized = torch.matmul(encodings,self.embedding.weight).view(inputs_shape)


#         e_latent_loss = F.mse_loss(quantized.detach(),inputs)
#         q_latent_loss = F.mse_loss(quantized,inputs.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss

#         quantized = inputs + (quantized - inputs).detach()
#         quantized = quantized.permute(0, 3, 1, 2).contiguous()

#         avg_probs = torch.mean(encodings,dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         return {'distances': distance,
#                 'quantize': quantized,
#                 'loss': loss,
#                 'encodings': encodings,
#                 'encoding_indices': encoding_indices,
#                 'perplexity': perplexity
#                 }


# class VQVAE(nn.Module):
#     def __init__(self, encoder, decoder, quantizer, pre_vq_conv1,
#                 data_variance, name=None):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.quantizer = quantizer
#         self.pre_vq_conv1 = pre_vq_conv1
#         self.data_variance = data_variance

#     def forward(self, inputs):
#         z = self.pre_vq_conv1(self.encoder(inputs))


#         vq_output = self.quantizer(z)

#         x_reconstructed = self.decoder(vq_output['quantize'])
#         reconstructed_error = torch.mean(torch.square(x_reconstructed - inputs) / self.data_variance)



#         loss = reconstructed_error + vq_output['loss']
#         return {
#             'z': z,
#             'x_reconstructed': x_reconstructed,
#             'loss': loss,
#             'reconstructed_error': reconstructed_error,
#             'VQ_error': vq_output['loss'],
#             'vq_output': vq_output
#         }