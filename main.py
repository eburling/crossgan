""" Cross-domain learning model (WIP)
"""

import argparse
import os

from models import PhenoVAE

parser = argparse.ArgumentParser(description='')

parser.add_argument('--data_dir',       type=str,   default='data',     help='input data directory')
parser.add_argument('--save_dir',       type=str,   default='save',     help='save directory')
parser.add_argument('--phase',          type=str,   default='train',    help='train or load')

parser.add_argument('--image_size',     type=int,   default=64,         help='image size')
parser.add_argument('--image_channel',  type=int,   default=3,          help='image channels')

parser.add_argument('--latent_dim',     type=int,   default=2,          help='latent dimension')
parser.add_argument('--inter_dim',      type=int,   default=64,         help='intermediate dimension')
parser.add_argument('--num_conv',       type=int,   default=3,          help='number of convolutions')
parser.add_argument('--batch_size',     type=int,   default=32,         help='batch size')
parser.add_argument('--epochs',         type=int,   default=2,          help='training epochs')
parser.add_argument('--nfilters',       type=int,   default=64,         help='num convolution filters')
parser.add_argument('--learn_rate',     type=float, default=0.001,      help='learning rate')
parser.add_argument('--epsilon_std',    type=float, default=1.0,        help='epsilon width')
parser.add_argument('--latent_samp',    type=int,   default=10,         help='number of latent samples')
parser.add_argument('--verbose',        type=int,   default=2,          help='1=verbose, 2=quiet')

args = parser.parse_args()


def main():

    os.makedirs(args.save_dir, exist_ok=True)
        
    if args.phase == 'train':
        model = PhenoVAE(args)
        model.train()

    if args.phase == 'load':
        from keras.models import load_model
        model = load_model(os.path.join(args.save_dir, 'vae_model.h5'))
        
    
if __name__ == '__main__':
    main()
