"""
Train a GAN to hide messages in images
"""

from SecretGAN import SecretGAN
import argparse
import time
import json
import numpy as np


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
    print(args.lr, args.epochs)
    #
    # Load dataset
    #
    print('\n...Loading training data ...\n')
    net.load_mnist()
    #
    # Train the GAN
    #
    print('...Training ...\n')
    start = time.time()
    net.train()
    end = time.time()
    #
    # Save the generator and discriminator
    #
    print('...Saving networks and training data...')
    net.save(args.save_dir+'discriminator', args.save_dir+'generator')

    output_dict = {
        'loss_generator': net.logger['loss_g'],
        'loss_discriminator': net.logger['loss_d'],
        'training_time': end-start,
    }
    with open(args.save_dir+'training_info.json', 'w') as fp:
        json.dump(output_dict, fp)
    #
    # Plot the training data
    #
    net.plot_training()