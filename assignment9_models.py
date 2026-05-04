"""Model definitions for Assignment 9.

The notebook acts as the main program. This module keeps the neural-network
architectures reusable and compact.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.log_var(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        recon = self.decoder(z)
        return recon, mean, log_var


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim, img_shape=(1, 70, 70), base_channels=16):
        super().__init__()
        channels = img_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *img_shape)
            feature_shape = self.features(dummy).shape[1:]
            flat_dim = math.prod(feature_shape)
        self.mean = nn.Linear(flat_dim, latent_dim)
        self.log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.features(x).view(x.size(0), -1)
        return self.mean(h), self.log_var(h)


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, img_shape=(1, 70, 70), base_channels=16, start_size=5):
        super().__init__()
        self.base_channels = base_channels
        self.start_size = start_size
        self.output_size = img_shape[-2:]
        self.fc = nn.Linear(latent_dim, base_channels * 8 * start_size * start_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base_channels // 2, img_shape[0], kernel_size=3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.base_channels * 8, self.start_size, self.start_size)
        h = self.net(h)
        h = F.interpolate(h, size=self.output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(h))


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, math.prod(img_shape)),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class MLPDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(math.prod(img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))


class CNNGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape=(1, 70, 70), base_channels=16, start_size=5):
        super().__init__()
        self.base_channels = base_channels
        self.start_size = start_size
        self.output_size = img_shape[-2:]
        self.fc = nn.Linear(latent_dim, base_channels * 8 * start_size * start_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base_channels // 2, img_shape[0], kernel_size=3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.base_channels * 8, self.start_size, self.start_size)
        h = self.net(h)
        h = F.interpolate(h, size=self.output_size, mode="bilinear", align_corners=False)
        return torch.tanh(self.out(h))


class CNNDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 70, 70), base_channels=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_shape[0], base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, *img_shape)).view(1, -1).size(1)
        self.classifier = nn.Sequential(nn.Linear(flat_dim, 1), nn.Sigmoid())

    def forward(self, img):
        h = self.features(img).view(img.size(0), -1)
        return self.classifier(h)
