import os
import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

class VGGGenerator(nn.Sequential):
    def __init__(self, latent_dim: int = 1024, output_channels: int = 3):
        super(VGGGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=7*7*256),
            nn.ReLU(), nn.BatchNorm1d(7*7*256)
        )
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 14, 14)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 14, 14)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 28, 28)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 28, 28)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=128, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # output shape: (256, 56, 56)
            nn.ConvTranspose2d(kernel_size=3, in_channels=128, out_channels=64, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # output shape: (256, 112, 112)
            nn.ConvTranspose2d(kernel_size=3, in_channels=64, out_channels=output_channels, stride=1, padding=1), # output shape: (256, 112, 112)
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0),256,7,7)
        out = self.convs(out)
        return out

class VGGMnistGenerator(nn.Sequential):
    def __init__(self, latent_dim: int = 1024, output_channels: int = 3):
        super(VGGMnistGenerator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=1*1*256),
            nn.ReLU(), nn.BatchNorm1d(1*1*256)
        )
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 3, 3)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 3, 3)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 7, 7)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # output shape: (256, 7, 7)
            nn.ConvTranspose2d(kernel_size=3, in_channels=256, out_channels=128, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # output shape: (256, 14, 14)
            nn.ConvTranspose2d(kernel_size=3, in_channels=128, out_channels=64, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # output shape: (256, 25, 28)
            nn.ConvTranspose2d(kernel_size=3, in_channels=64, out_channels=output_channels, stride=1, padding=1), # output shape: (256, 112, 112)
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0),256,1,1)
        out = self.convs(out)
        return out


class VGGDiscriminator(nn.Module):
    def __init__(self, input_channels: int = 3, alpha: float = 0.2):
        super(VGGDiscriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=input_channels, out_channels=64, stride=2, padding=2),
            nn.LeakyReLU(alpha), # output shape: (64, 56, 56)
            nn.Conv2d(kernel_size=5, in_channels=64, out_channels=128, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha), # output shape: (128, 28, 28)
            nn.Conv2d(kernel_size=5, in_channels=128, out_channels=256, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha), # output shape: (256, 14, 14)
            nn.Conv2d(kernel_size=5, in_channels=256, out_channels=512, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha), # output shape: (512, 7, 7)
        )
        self.fc = nn.Linear(in_features=512*7*7, out_features=1)
    
    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512*7*7)
        out = self.fc(out)
        return out


class VGGMnistDiscriminator(nn.Module):
    def __init__(self, input_channels: int = 3, alpha: float = 0.2):
        super(VGGMnistDiscriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=input_channels, out_channels=64, stride=2, padding=2),
            nn.LeakyReLU(alpha), # output shape: (64, 14, 14)
            nn.Conv2d(kernel_size=5, in_channels=64, out_channels=128, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha), # output shape: (128, 7, 7)
            nn.Conv2d(kernel_size=5, in_channels=128, out_channels=256, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha), # output shape: (256, 4, 4)
            nn.Conv2d(kernel_size=5, in_channels=256, out_channels=512, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha), # output shape: (512, 2, 2)
        )
        self.fc = nn.Linear(in_features=512*2*2, out_features=1)
    
    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512*2*2)
        out = self.fc(out)
        return out
