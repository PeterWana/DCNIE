from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras import metrics
import numpy as np
from sklearn import decomposition
from keras import optimizers
import keras
import keras.layers.advanced_activations
from tensorflow import optimizers
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd

def sampling(args):
    z_mean, z_log_var = args
    epsilon_std = 1.0
    batchl = K.shape(z_mean)[0]
    diml = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batchl, diml), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def variational_auto_encoder(x, dim, latent_dim):

    dim_len = len(dim)
    if dim_len == 0:
        return x

    rows = x.shape[0]
    dim_input = rows
    input_matrix = Input(shape=(dim_input,))

    activation = 'softsign'
    h = Dense(dim[0], activation=activation)(input_matrix)
    for i in range(dim_len):
        if i > 0:
            h = Dense(dim[i], activation=activation)(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    if dim_len == 1:
        decoder_h = Dense(dim[0], activation=activation)(z)
        x_decoded_mean = Dense(dim_input, activation='softplus')(decoder_h)

    if dim_len > 1:
        decoder_h = Dense(dim[dim_len - 2], activation=activation)(z)
        for i in range(dim_len):
            if i > 1:
                decoder_h = Dense(dim[dim_len - 1 - i], activation=activation)(decoder_h)
        x_decoded_mean = Dense(dim_input, activation='softplus')(decoder_h)

    vae = Model(input_matrix, x_decoded_mean)
    encoder = Model(input_matrix, z_mean)
    xent_loss = dim_input * metrics.categorical_crossentropy(input_matrix, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)
    optimizer = optimizers.Adamax(learning_rate=0.014, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.00)

    vae.compile(optimizer=optimizer,  metrics=['accuracy'])
    vae.summary()

    x_train = x
    try:
        x_train = x_train.values.reshape((len(x_train), np.prod(x_train.shape[1:])))
    except:
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    vae.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True)

    res = encoder.predict(x_train)
    return res




