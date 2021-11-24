"""
Networks used for generative adversarial networks
"""

import torch
from torch import nn

torch.manual_seed(10)


class DiscriminatorNetwork(nn.Module):
    """
    Discriminator network to distinguish real and fake images.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # flatten the input image
        x = x.view(x.size(0), 784)
        output = self.model(x)
        
        return output


class GeneratorSecretNetwork(nn.Module):
    """
    Generator Network to hide messages in images. Used in SecretGAN.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        # reshape the image
        output = output.view(x.size(0), 1, 28, 28)
        
        return output


class GeneratorNetwork(nn.Module):
    """
    Generator network to create images from latent space
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        # reshape the image
        output = output.view(x.size(0), 1, 28, 28)
        
        return output

