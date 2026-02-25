import os
import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

from src.model import VGGMnistGenerator, VGGMnistDiscriminator
from src.train import train_least_gan

if __name__=="__main__":

    batch_size = 128
    transform = T.Compose([
        T.ToTensor(), T.Normalize((0.5,),(0.5,))
    ])
    data = torchvision.datasets.MNIST(
        root="./mnist", train=True, download=True, transform=transform)
    print(f"Dataset size: {len(data)}")

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    lr_D = 0.0002
    lr_G = 0.0002
    latent_dim = 1024
    fixed_noise = torch.normal(0., 1., (10, latent_dim))

    a = 0.
    b = 1.
    c = 1.

    generator = VGGMnistGenerator(latent_dim=latent_dim, output_channels=1)
    discriminator = VGGMnistDiscriminator(input_channels=1, alpha=0.2)
    train_least_gan(D=discriminator, G=generator, device=device,
                    lr_D=lr_D, lr_G=lr_G, a=a, b=b, c=c,
                    num_epochs=num_epochs, latent_dim=latent_dim,
                    dataloader=dataloader,
                    fixed_noise=fixed_noise)

