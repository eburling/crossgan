import os
import csv
import random
import glob 
import imageio

from PIL import Image
import numpy as np
from scipy.stats import norm

from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose
from keras import optimizers
from keras import metrics
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


class PhenoVAE():
    """ 2-dimensional variational autoencoder
    """
    
    def __init__(self, args):
        """ initialize model with argument parameters and build
        """

        self.data_dir       = args.data_dir
        self.save_dir       = args.save_dir
        self.image_size     = args.image_size
        self.image_channel  = args.image_channel
        self.latent_dim     = args.latent_dim
        self.inter_dim      = args.inter_dim
        self.num_conv       = args.num_conv
        self.batch_size     = args.batch_size
        self.epochs         = args.epochs
        self.nfilters       = args.nfilters
        self.learn_rate     = args.learn_rate
        self.epsilon_std    = args.epsilon_std
        self.latent_samp    = args.latent_samp
        self.verbose        = args.verbose
        
        self.phase          = args.phase
        
        self.imageList      = glob.glob(os.path.join(self.data_dir, '*.png'))
        
        self.build_model()


    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """
        
        z_mean, z_log_var = sample_args
        
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                  mean=0,
                                  stddev=self.epsilon_std)
    
        return z_mean + K.exp(z_log_var) * epsilon
    
    
    def build_model(self):
        """ build VAE model
        """
        
        input_dim = (self.image_size, self.image_size, self.image_channel)
        
        #   encoder architecture
        
        x = Input(shape=input_dim)
        
        conv_1 = Conv2D(self.image_channel,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu')(x)
        
        conv_2 = Conv2D(self.nfilters,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu',
                        strides=(2, 2))(conv_1)
        
        conv_3 = Conv2D(self.nfilters,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_2)
        
        conv_4 = Conv2D(self.nfilters,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_3)
        
        flat = Flatten()(conv_4)
        hidden = Dense(self.inter_dim, activation='relu')(flat)
        
        #   reparameterization trick
        
        z_mean = Dense(self.latent_dim)(hidden)        
        z_log_var = Dense(self.latent_dim)(hidden)
        
        z = Lambda(self.sampling)([z_mean, z_log_var])
        
        
        #   decoder architecture

        output_dim = (self.batch_size, self.image_size//2, self.image_size//2, self.nfilters)
        
        #   instantiate rather than pass through 
        
        decoder_hid = Dense(self.inter_dim, 
                            activation='relu')
        
        decoder_upsample = Dense(self.nfilters * self.image_size//2 * self.image_size//2, 
                                 activation='relu')

        decoder_reshape = Reshape(output_dim[1:])
        
        decoder_deconv_1 = Conv2DTranspose(self.nfilters,
                                           kernel_size=self.num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        
        decoder_deconv_2 = Conv2DTranspose(self.nfilters,
                                   kernel_size=self.num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
        
        decoder_deconv_3_upsamp = Conv2DTranspose(self.nfilters,
                                                  kernel_size = self.num_conv,
                                                  strides = (2, 2),
                                                  padding = 'valid',
                                                  activation = 'relu')
        
        decoder_mean_squash = Conv2D(self.image_channel,
                                     kernel_size = 2,
                                     padding = 'valid',
                                     activation = 'sigmoid')
        
        hid_decoded             = decoder_hid(z)
        up_decoded              = decoder_upsample(hid_decoded)
        reshape_decoded         = decoder_reshape(up_decoded)
        deconv_1_decoded        = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded        = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu          = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash   = decoder_mean_squash(x_decoded_relu)

        #   need to keep generator model separate so new inputs can be used
        
        decoder_input           = Input(shape=(self.latent_dim,))
        _hid_decoded            = decoder_hid(decoder_input)
        _up_decoded             = decoder_upsample(_hid_decoded)
        _reshape_decoded        = decoder_reshape(_up_decoded)
        _deconv_1_decoded       = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded       = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu         = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash  = decoder_mean_squash(_x_decoded_relu)
        
        
        #   instantiate VAE models
        
        self.vae = Model(x, x_decoded_mean_squash)
        self.encoder = Model(x, z_mean)
        self.decoder = Model(decoder_input, _x_decoded_mean_squash)
        
        
        #   VAE loss terms w/ KL divergence
            
        def vae_loss(x, x_decoded_mean_squash):
            xent_loss = self.image_size * self.image_size * metrics.binary_crossentropy(K.flatten(x),
                                                                                        K.flatten(x_decoded_mean_squash))
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(xent_loss + kl_loss)
            return vae_loss
        
        
        
#        xent_loss = self.image_size * self.image_size * metrics.binary_crossentropy(K.flatten(x), 
#                                                                                    K.flatten(x_decoded_mean_squash))
#        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#        vae_loss = K.mean(xent_loss + kl_loss)
#        self.vae.add_loss(vae_loss)
        
        rms = optimizers.adagrad(lr = self.learn_rate)
        
        self.vae.compile(optimizer = rms,
                         loss = vae_loss)

        
        self.vae.summary()
            
    
    def train(self):
        """ train VAE model
        """

        datagen = ImageDataGenerator(rescale = 1./255,
                                           horizontal_flip = True,
                                           vertical_flip = True)
                        
        train_generator = datagen.flow_from_directory(
                self.data_dir,
                target_size = (self.image_size, self.image_size),
                batch_size = self.batch_size,
                class_mode = 'input')
        
        self.vae.fit_generator(train_generator,
                               epochs = self.epochs,
                               steps_per_epoch = 47000 // self.batch_size)
                
        self.vae.save(os.path.join(self.save_dir, 'vae_model.h5'))

        self.encode()        
        self.latent_walk()
        self.save_in_out()
        
        
    def save_in_out(self):
        print('save samples of input images and reconstructions')

    
    def latent_walk(self):
        """ latent space walking
        """
        
        figure = np.zeros((self.image_size * self.latent_dim, self.image_size * self.latent_samp, self.image_channel))
        
        grid_x = norm.ppf(np.linspace(0.05, 0.95, self.latent_samp))
        
        for i in range(self.latent_dim):
            for j, xi in enumerate(grid_x):
                z_sample = np.zeros(self.latent_dim)
                z_sample[i] = xi
                # since model expects a certain batch_size, feed identical samples
                z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, self.latent_dim)
                x_decoded = self.decoder.predict(z_sample, batch_size=self.batch_size)
                
                sample = x_decoded[0].reshape(self.image_size, self.image_size, self.image_channel)
                
                figure[i * self.image_size: (i + 1) * self.image_size,
                       j * self.image_size: (j + 1) * self.image_size, :] = sample
        
        imageio.imwrite(os.path.join(self.save_dir, 'latent_walk.png'), figure)
    
    
    def encode(self):
        """ encode data with trained model
        """
        
        x_test_encoded = self.encoder.predict(self.loadedImages, 
                                              batch_size=self.batch_size)
        
        outFile = open(os.path.join(self.save_dir, 'encodings.csv'), 'w')
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(x_test_encoded)
        
        
        
        
        
