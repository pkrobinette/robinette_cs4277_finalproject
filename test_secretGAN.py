"""
Load a pretrained GAN and generate images
"""

from SecretGAN import SecretGAN
import argparse
import time
import json


def get_args():
    """
    Parse the command arguments
    """
    # create a parser
    parser = argparse.ArgumentParser()
    # Add the agent argument to the parser
    parser.add_argument("--lr", type=float, default= 0.0001, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--save_dir', type=str, default='', help='path to save the trained policy')
    
    # execute the parser
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    #
    # Set up GAN to train
    #
    net = SecretGAN(lr=args.lr, batch_size=args.batch_size, epochs=args.epochs)
    #
    # Load dataset
    #
    print('\n...Loading training data ...\n')
    net.load_mnist()
    #
    # Load trained GAN
    #
    print('...Loading trained GAN ...\n')
    net.load('saved_models/discriminator', 'saved_models/generator')
    #
    # Generating examples
    #
    net.compare_images()
    #
    # Plot loss curves from training
    #
    with open('saved_models/training_info.json') as f:
        data = json.load(f)
    
    net.logger['loss_d'] = data['loss_discriminator']
    net.logger['loss_g'] = data['loss_generator']

    net.plot_training()
