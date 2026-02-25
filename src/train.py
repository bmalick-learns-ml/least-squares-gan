import os
import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


def plot_images(images: torch.Tensor, nrow: int = 8,
                figsize: Tuple[float] = (8,6), save_name: str = None,
                show: bool = True, title: str = None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(T.ToPILImage()(torchvision.utils.make_grid(images, nrow=nrow)))
    ax.axis("off")
    if title: ax.set_title(title)
    if save_name: fig.savefig(save_name, bbox_inches="tight", pad_inches=1)
    if show: plt.show()
    plt.close()

def update_discriminator(x, z, G, D, trainer_D, a, b):
    criterion = nn.MSELoss()
    trainer_D.zero_grad()
    bs = x.size(0)
    real_labels = torch.zeros((bs,), device=x.device).fill_(b)
    fake_labels = torch.zeros((bs,), device=x.device).fill_(a)
    fake_x = G(z)
    fake_y = D(fake_x.detach())
    real_y = D(x)
    loss_D = 0.5 * criterion(real_y, real_labels.reshape(real_y.shape)) + 0.5 * criterion(fake_y, fake_labels.reshape(fake_y.shape))
    loss_D.backward()
    trainer_D.step()
    return loss_D.item()

def update_generator(z, G, D, trainer_G, c):
    criterion = nn.MSELoss()
    trainer_G.zero_grad()
    bs = z.size(0)
    real_labels = torch.zeros((bs,), device=z.device).fill_(c)
    fake_y = D(G(z))
    loss_G = 0.5 * criterion(fake_y, real_labels.reshape(fake_y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G.item()


def init_weights(w):
    if isinstance(w, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(w.weight.data, 0., 0.02)
        if w.bias is not None:
            torch.nn.init.constant_(w.bias.data, 0.0)
    elif isinstance(w, (nn.BatchNorm2d, nn.BatchNorm1d)):
        torch.nn.init.normal_(w.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(w.bias.data, 0.0)

def train_least_gan(D, G, device, lr_D, lr_G, a, b, c, num_epochs, latent_dim, fixed_noise, dataloader):
    print(f"Device: {device}")
    D = D.to(device)
    G = G.to(device)
    fixed_noise = fixed_noise.to(device)

    # for w in D.parameters(): torch.nn.init.normal_(w, 0., 0.02)
    # for w in G.parameters(): torch.nn.init.normal_(w, 0., 0.02)

    D.apply(init_weights)
    G.apply(init_weights)

    trainer_D = torch.optim.Adam(D.parameters(), lr_D, betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(G.parameters(), lr_G, betas=(0.5, 0.999))

    metrics = []
    os.makedirs("visualizations", exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss_D = 0.
        epoch_loss_G = 0.
        num_instances = 0.

        for batch in dataloader:
            x, _ = batch
            bs = x.size(0)
            x = x.to(device)
            z = torch.normal(0., 1., (bs, latent_dim), device=device)
            loss_D = update_discriminator(x=x, z=z, G=G, D=D, a=a, b=b, trainer_D=trainer_D)
            loss_G = update_generator(z=z, G=G, D=D, c=c, trainer_G=trainer_G)
            epoch_loss_D += loss_D * bs
            epoch_loss_G += loss_G * bs
            num_instances += bs
        epoch_loss_D /= num_instances
        epoch_loss_G /= num_instances
        metrics.append([epoch_loss_D, epoch_loss_G])
        print(f"[Epoch {epoch}/{num_epochs}] loss_D: {epoch_loss_D:.4f}, loss_G: {epoch_loss_G:.4f}")
    
        gen_imgs = G(fixed_noise)
        plot_images(gen_imgs, nrow=5, figsize=(19.2,10.8), save_name=f"visualizations/{epoch:2d}.png", show=False, title=f"Epoch {epoch:02d}")

    metrics = np.array(metrics)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(metrics[:, 0], label="loss_D")
    ax.plot(metrics[:, 1], label="loss_G")
    ax.legend()
    fig.savefig("losses.png")
    plt.show()
    plt.close()

    
