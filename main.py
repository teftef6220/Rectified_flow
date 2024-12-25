import os
import time

import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import wandb

from unet import UNet
from embeddings import SinusoidalTimestepEmbedding, ClassEmbedding

epochs = 200

wandb.init(
    project="mnist-rectifiedflow",
    config={
        "epochs": epochs,
    }
)

transform = transforms.Compose(
    [transforms.ToTensor(),]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = th.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = th.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

unet = UNet(3, embedding_channels=256, cond_embed_dim=2048, time_embed_dim=2048, depth=4, kernel_size=[3,3,3,3,3,3,3], layers=[3,3,3,6,3,3,3], num_groups=[8,8,8,8,8,8,8])
time_embed = SinusoidalTimestepEmbedding(2048)
cond_embed = ClassEmbedding(10, 2048)

model = nn.ModuleList([unet, time_embed, cond_embed]).cuda()

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# time as folder name
folder_name = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(folder_name, exist_ok=True)

for epoch in range(epochs):
    with th.no_grad():
        model.eval()
        x_0 = th.randn(10, 3, 32, 32).cuda()
        time_embedded = time_embed(th.linspace(0, 1, 10).cuda())
        cond_embedded = cond_embed(nn.functional.one_hot(th.arange(10), 10).float().cuda())
        for i in range(10):
            v = unet(x_0, time_embedded[i], cond_embedded)
            x_0 = x_0 + 0.1 * v
        sample = (x_0 + 1) / 2
        sample.clamp_(0, 1)
        wandb.log({"sample": [wandb.Image(x) for x in sample]})

    th.save(model.state_dict(), f"{folder_name}/epoch_{epoch}.pt")

    model.train()
    wandb.log({"epoch": epoch})
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs * 2 - 1
        labels = nn.functional.one_hot(labels, 10).float()
        x_0 = th.randn_like(inputs)
        t = 1 - th.rand(1).cuda()
        x_t = t * x_0 + (1 - t) * inputs

        time_embedded = time_embed(t)
        cond_embedded = cond_embed(labels)

        v = unet(x_t, time_embedded, cond_embedded)
        loss = nn.functional.mse_loss(inputs - x_0, v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss})
        print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")
print('Finished Training')