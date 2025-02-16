import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from VQVAE.my_data_loader import MyDataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from argparse import ArgumentParser


# parser
parser = ArgumentParser()
parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer")
args = parser.parse_args()


# ---------------------------
# エンコーダー：可変サイズ対応
# ---------------------------
class ConvVAE_Encoder(nn.Module):
    def __init__(self, latent_dim=26):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)   # 64 -> 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 16 -> 8
        self.relu = nn.ReLU()
        # 入力は [B, 128, 8, 8] となるので、flatten 後のサイズは 128*8*8=8192
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # [B, 32, 32, 32]
        x = self.relu(self.conv2(x))  # [B, 64, 16, 16]
        x = self.relu(self.conv3(x))  # [B, 128, 8, 8]
        x = x.view(x.size(0), -1)     # [B, 8192]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma
# ---------------------------
# デコーダー：可変出力対応
# ---------------------------
class ConvVAE_Decoder(nn.Module):
    def __init__(self, latent_dim=26):
        super().__init__()
        # latent から 128×8×8 の初期特徴マップに変換
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.relu = nn.ReLU()
        # 逆畳み込みでアップサンプリング（8->16->32->64）
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 8->16
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 16->32
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # 32->64
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z)         # [B, 128*8*8]
        x = self.relu(x)
        x = x.view(-1, 128, 8, 8)  # [B, 128, 8, 8]
        x = self.relu(self.deconv1(x))  # [B, 64, 16, 16]
        x = self.relu(self.deconv2(x))  # [B, 32, 32, 32]
        x = self.sigmoid(self.deconv3(x))  # [B, 3, 64, 64]
        return x

# ---------------------------
# 再パラメータ化
# ---------------------------
def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    return mu + sigma * eps

# ---------------------------
# VAE本体
# ---------------------------
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=26):
        super().__init__()
        self.encoder = ConvVAE_Encoder(latent_dim=latent_dim)
        self.decoder = ConvVAE_Decoder(latent_dim=latent_dim)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        recon_x = self.decoder(z)
        return recon_x, mu, sigma
    

# ---------------------------
# KL アニーリング　っていうらしい
# ---------------------------
def kl_weight(epoch, warmup_epochs=10):
    return min(1.0, epoch / warmup_epochs)

if __name__ == '__main__':
    # パラメータ設定
    latent_dim = 26
    batch_size = 16
    epochs = 100
    lr = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_every_n_epochs = 50

    # データセットの読み込み
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = MyDataLoader(data_dir='./make_dataset/stamps', transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    if args.optimizer == "RAdamScheduleFree":
        from schedulefree import RAdamScheduleFree
        optimizer = RAdamScheduleFree(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)


    # エポック数とロスを記録するリスト
    epoch_vec = []
    total_KL_loss_array = []
    total_recon_loss_array = []

    for epoch in range(epochs):
        model.train()
        if args.optimizer == "RAdamScheduleFree":
            optimizer.train()
        total_loss = []
        total_mse = []
        total_kld = []

        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon_imgs, mu, sigma = model(imgs)
            mse_loss = F.mse_loss(recon_imgs, imgs, reduction='sum')
            kld_loss = - torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
            loss = (mse_loss + kl_weight(epoch) * kld_loss) / imgs.size(0)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            total_mse.append(mse_loss.item())
            total_kld.append(kld_loss.item())

        ave_loss = sum(total_loss) / len(total_loss)
        ave_mse = sum(total_mse) / len(total_loss)
        ave_kld = sum(total_kld) / len(total_loss)
        print(f"Epoch {epoch+1}: Loss {ave_loss:.3f} | MSE {ave_mse:.3f} | KLD {ave_kld:.3f}")

        # 学習経過のロスを更新してプロット保存
        epoch_vec.append(epoch+1)
        total_KL_loss_array.append(ave_kld)
        plt.figure()
        plt.plot(epoch_vec, total_KL_loss_array)
        plt.xlabel("Epoch")
        plt.ylabel("KL Loss")
        plt.title("Training KL Loss")
        plt.savefig(f"./VAE/result/KL_loss_plot.png")
        plt.close()

        total_recon_loss_array.append(ave_mse)
        plt.figure()
        plt.plot(epoch_vec, total_recon_loss_array)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Recon Loss")
        plt.savefig(f"./VAE/result/recon_loss_plot.png")
        plt.close()

        # 定期的にモデルと再構成画像を保存
        if (epoch + 1) % save_every_n_epochs == 0:
            print(f"Saving model at epoch {epoch+1}")
            torch.save(model.state_dict(), f"./VAE/model/ConvVAE_{epoch+1}.pth")
        
        with torch.no_grad():
            model.eval()
            if args.optimizer == "RAdamScheduleFree":
                optimizer.eval()
            z = torch.randn(64, latent_dim).to(device)
            samples = model.decoder(z)
            samples = samples * 0.5 + 0.5
            samples = samples.clamp(0.0, 1.0)  
            grid = torchvision.utils.make_grid(samples, nrow=8)
            torchvision.utils.save_image(grid, f"./VAE/result/epoch_{epoch+1}.png")
