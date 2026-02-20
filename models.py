import torch
import torch.nn as nn


class Generator(nn.Module):
    """Simple DCGAN-style generator for MNIST: z -> (1,28,28)"""

    def __init__(self, z_dim: int = 128, img_channels: int = 1, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (N, z_dim, 1, 1) -> (N, 256, 7, 7)
            nn.ConvTranspose2d(z_dim, feature_maps * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # (N, 256, 7, 7) -> (N, 128, 14, 14)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # (N, 128, 14, 14) -> (N, 1, 28, 28)
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN-style discriminator for MNIST: (1,28,28) -> score + feature representation"""

    def __init__(self, img_channels: int = 1, feature_maps: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            # (N, 1, 28, 28) -> (N, 64, 14, 14)
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 64, 14, 14) -> (N, 128, 7, 7)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            # (N, 128, 7, 7) -> (N, 1, 1, 1)
            nn.Conv2d(feature_maps * 2, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor):
        feat = self.features(x)            # (N, 128, 7, 7)
        out = self.classifier(feat)        # (N, 1, 1, 1)
        out = out.view(x.size(0))          # (N,)
        return out, feat                   # return score + features


class Encoder(nn.Module):
    """Encoder for f-AnoGAN: (1,28,28) -> z"""

    def __init__(self, z_dim: int = 128, img_channels: int = 1, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (N, 1, 28, 28) -> (N, 64, 14, 14)
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 64, 14, 14) -> (N, 128, 7, 7)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 128, 7, 7) -> (N, z_dim, 1, 1)
            nn.Conv2d(feature_maps * 2, z_dim, 7, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)  # (N, z_dim, 1, 1)
        return z
