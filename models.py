import os
import glob 
from PIL import Image
import numpy as np
        
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
        
        self.phase          = args.phase
        
        self.build_model()
        self.build_datagen()
    
    
    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """
        
        z_mean, z_log_var = sample_args
        
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                  mean=0,
                                  stddev=self.epsilon_std)
    
        return z_mean + K.exp(z_log_var) * epsilon
    
    
    def build_model(self):
        """ Build VAE model
        """
        
        input_dim = (self.image_size, self.image_size, self.image_channel)
        
        #   encoder architecture
        
        x = Input(shape=input_dim)
        
        conv_1 = Conv2D(self.image_channel,
                        kernel_size=(2, 2),
                        padding='same', activation='relu')(x)
        
        conv_2 = Conv2D(self.nfilters,
                        kernel_size=(2, 2),
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
                                                  kernel_size=(3, 3),
                                                  strides=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        
        decoder_mean_squash = Conv2D(self.image_channel,
                                     kernel_size=2,
                                     padding='valid',
                                     activation='sigmoid')
        
        hid_decoded             = decoder_hid(z)
        up_decoded              = decoder_upsample(hid_decoded)
        reshape_decoded         = decoder_reshape(up_decoded)
        deconv_1_decoded        = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded        = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu          = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash   = decoder_mean_squash(x_decoded_relu)

        #   need to keep generator model separate so new inputs can be used
        
        decoder_input = Input(shape=(self.latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
        
        
        #   instantiate VAE models
        
        self.vae = Model(x, x_decoded_mean_squash)
        self.encoder = Model(x, z_mean)
        self.decoder = Model(decoder_input, _x_decoded_mean_squash)
        
        
        #   VAE loss terms w/ KL divergence
        
        xent_loss = self.image_size * self.image_size * metrics.binary_crossentropy(K.flatten(x), 
                                                                                    K.flatten(x_decoded_mean_squash))
        
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        vae_loss = K.mean(xent_loss + kl_loss)
        
        self.vae.add_loss(vae_loss)
        
        rms = optimizers.adagrad(lr=self.learn_rate)
        
        self.vae.compile(optimizer=rms)
        
        self.vae.summary()
            
    
    def build_datagen(self):
        """ data generator for VAE
        """

        imageList=glob.glob(os.path.join(self.data_dir, '*.png'))
        
        self.loadedImages = np.array([np.array(Image.open(fname)) for fname in imageList])
        self.loadedImages = self.loadedImages.astype('float32') / 255.
        
        self.datagen = ImageDataGenerator(rescale=1. / 255,
                                     horizontal_flip = True,
                                     vertical_flip = True)
        
        self.datagen.fit(self.loadedImages)
        
        
    def train(self):
        """ train VAE model
        """
        
        self.vae.fit(self.loadedImages,
                     epochs = self.epochs,
                     batch_size = self.batch_size)
        
        self.vae.save(os.path.join(self.save_dir, 'vae_model.h5'))
        
#        self.vae.fit_generator(self.datagen.flow(self.loadedImages,
#                                                 None,
#                                                 batch_size = self.batch_size),
#                               shuffle=True,
#                               epochs=self.epochs,
#                               steps_per_epoch=100)
    
    def latent_walk(self):
        """ latent space walking
        """
        
        from scipy.stats import norm
        import matplotlib.pyplot as plt

        n = 15  # figure with 15x15 samples
        sample_size = self.image_size
        
        figure = np.zeros((sample_size * n, sample_size * n, self.image_channel))
        
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        
        
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, 2)
                x_decoded = self.decoder.predict(z_sample, batch_size=self.batch_size)
                sample = x_decoded[0].reshape(sample_size, sample_size, self.image_channel)
                figure[i * sample_size: (i + 1) * sample_size,
                       j * sample_size: (j + 1) * sample_size,:] = sample
        
        plt.figure(figsize=(10, 10))
        plt.savefig(os.path.join(self.save_dir, 'latent_walk.pdf'))
        #plt.imshow(figure, cmap='Greys_r')
        #plt.show()
    
    
    def encode(self):
        """ use a trained model to encode data set
        """
        
        
        
        
        
        
        