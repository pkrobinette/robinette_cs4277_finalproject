"""
A Generative Adversarial Network
"""
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from Networks import DiscriminatorNetwork, GeneratorSecretNetwork

import math
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(111)

class SecretGAN():
    """
    A GAN trained to hide messages in images
    """
    def __init__(self, lr=0.0001, batch_size= 32, epochs=50):
        # Parameters
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = None
        self.logger = {
            'loss_d': [],
            'loss_g': []
        }
        
        # Networks
        self.generator = GeneratorSecretNetwork()
        self.discriminator = DiscriminatorNetwork()
        
        # Optimizers
        self.loss_func = nn.BCELoss()
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        
        
    def load_mnist(self):
        """
        Load the MNIST dataset
        """
        compose = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        train_set = torchvision.datasets.MNIST(
            root=".", 
            train=True, 
            download=True, 
            transform=compose
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        
        
    
    def train(self):
        """
        Train the GAN
        """
        for epoch in range(self.epochs):
            for i, (real_samples, labels) in enumerate(self.train_loader):
                real_labels = torch.ones((self.batch_size, 1))
                generated_samples = self.generator(real_samples)
                generated_labels = torch.zeros((self.batch_size, 1))
                comb_samples = torch.cat((real_samples, generated_samples))
                comb_labels = torch.cat((real_labels, generated_labels))
                
                # train the discriminator
                out_d = self.discriminator(comb_samples)
                self.discriminator.zero_grad()
                loss_d = self.loss_func(out_d, comb_labels)
                loss_d.backward()
                self.discriminator_optimizer.step()
                
                # train the generator
                hidden_msgs = self.generator(real_samples)
                msgs_detected = self.discriminator(hidden_msgs)
                self.generator.zero_grad()
                loss_g = self.loss_func(msgs_detected, real_labels)
                loss_g.backward()
                self.generator_optimizer.step()
                
                # log data
                if i == self.batch_size-1:
                    print('Epoch: {} | Loss D: {} | Loss G: {}'.format(epoch, loss_d, loss_g))
                    self.logger['loss_d'].append(loss_d.detach().item())
                    self.logger['loss_g'].append(loss_g.detach().item())
                    
                    
    def generate_hidden_msg(self):
        """
        Test the generator
        """
        imgs, labels = next(iter(self.train_loader))
        hidden_msgs = self.generator(imgs)
        hidden_msgs = hidden_msgs.detach()

        return hidden_msgs
        
                    
    def show_images(self, samples=None, title=None):
        """
        Show real and generated images
        """
        if samples == None:
            samples, labels = next(iter(self.train_loader))
        for i in range(self.batch_size):
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(samples[i].reshape(28, 28), cmap="gray_r")
            plt.xticks([])
            plt.yticks([])
        if title != None:
            plt.title(title)
        plt.show()

    def compare_images(self, path=None):
        """
        Create a plot to compare real and generated images
        """
        real, labels = next(iter(self.train_loader))
        generated = self.generate_hidden_msg()

        plt.figure(figsize=(10, 8))
        
        for i in range(self.batch_size*2):
            if i%2 == 0:
                ax = plt.subplot(8, 8, i+1)
                ax.set_title(str(int(i/2))+") real:")  # set title
                plt.imshow(real[int(i/2)].reshape(28,28), cmap="gray_r")
                plt.xticks([])
                plt.yticks([])
  
                ax = plt.subplot(8, 8, i+2)
                ax.set_title(str(int(i/2))+") Secret:")
                plt.imshow(generated[int(i/2)].reshape(28,28), cmap="gray_r")
                plt.xticks([])
                plt.yticks([])
            
        plt.tight_layout()
        if path != None:
            plt.savefig(path+'.png')
        plt.show()
                    
                    
    def plot_training(self, path=None):
        """
        Plot the generator and discriminator loss
        """
        plt.figure(1)
        plt.plot(self.logger['loss_g'], label='Generator Loss')
        plt.plot(self.logger['loss_d'], label='Discriminator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Generator and Discriminator Loss')
        plt.legend()
        if path != None:
            plt.savefig(path+'.png')
        plt.show()
        
    
    def save(self, disc_path, gen_path):
        """
        Save the generator and discriminator
        """
        # save discriminator
        torch.save(self.discriminator, disc_path)
        # save generator
        torch.save(self.generator, gen_path)
        
    def load(self, disc_path, gen_path):
        """
        Load the generator and discriminator
        """
        # load discriminator
        self.discriminator = torch.load(disc_path)
        # load the generator
        self.generator = torch.load(gen_path)