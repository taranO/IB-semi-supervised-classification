import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Lambda
from keras.layers import Dropout, BatchNormalization
from keras.layers import Activation, ReLU, LeakyReLU, Softmax
from keras import backend as K

import sys

from libs.BaseModel import BaseModel
# ======================================================================================================================
class ModelEncoderBNLaten2Constraint(BaseModel):
    def __init__(self, layers_conv=[], is_bn=False, dim_a=1024):

        self.dim_a = dim_a
        self.layers = []
        if layers_conv:
            for filters in layers_conv:
                self.layers.append(Conv2D(filters, kernel_size=4, strides=2, padding='same'))
                if is_bn:
                    self.layers.append(BatchNormalization())

                self.layers.append(LeakyReLU(alpha=0.1))

        self.layers.append(Flatten())

        self.dense_out = keras.layers.Dense(units=dim_a)
        self.dense_vae_mean = keras.layers.Dense(units=dim_a)
        self.dense_vae_var  = keras.layers.Dense(units=dim_a)


    # init forward pass
    def init(self, inputs):

        n = len(self.layers)
        for i in range(n):
            if i == 0:
                x = self.layers[i](inputs)
            else:
                x = self.layers[i](x)

        laten_a       = self.dense_out(x)
        mean_a    = self.dense_vae_mean(laten_a)
        log_var_a = self.dense_vae_var(laten_a)

        return Model(inputs, [laten_a, mean_a, log_var_a])
# # v1
# class ModelEncoderBNLaten2Constraint(BaseModel):
#     def __init__(self, layers_conv=[], is_bn=False, dim_a=1024):
#
#         self.dim_a = dim_a
#         self.layers = []
#         if layers_conv:
#             for filters in layers_conv:
#                 self.layers.append(Conv2D(filters, kernel_size=4, strides=2, padding='same'))
#                 if is_bn:
#                     self.layers.append(BatchNormalization())
#
#                 self.layers.append(LeakyReLU(alpha=0.1))
#
#         self.layers.append(Flatten())
#
#         self.dense_out = keras.layers.Dense(units=dim_a)
#         self.dense_vae_mean = keras.layers.Dense(units=dim_a)
#         self.dense_vae_var  = keras.layers.Dense(units=dim_a)
#
#     # reparameterization trick
#     # instead of sampling from Q(z|X), sample epsilon = N(0,I)
#     # z = z_mean + sqrt(var) * epsilon
#     def sampling(self, args):
#         """Reparameterization trick by sampling
#             fr an isotropic unit Gaussian.
#         # Arguments:
#             args (tensor): mean and log of variance of Q(z|X)
#         # Returns:
#             z (tensor): sampled latent vector
#         """
#
#         z_mean, z_log_var = args
#         batch = K.shape(z_mean)[0]
#         dim = K.int_shape(z_mean)[1]
#         # by default, random_normal has mean=0 and std=1.0
#         epsilon = K.random_normal(shape=(batch, dim))
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon
#
#     # init forward pass
#     def init(self, inputs):
#
#         n = len(self.layers)
#         for i in range(n):
#             if i == 0:
#                 x = self.layers[i](inputs)
#             else:
#                 x = self.layers[i](x)
#
#         laten_a       = self.dense_out(x)
#         mean_a    = self.dense_vae_mean(laten_a)
#         log_var_a = self.dense_vae_var(laten_a)
#
#         # use reparameterization trick to push the sampling out as input
#         # with the TensorFlow backend
#         out_a = Lambda(self.sampling, output_shape=(self.dim_a,), name='a')([mean_a, log_var_a])
#
#         return Model(inputs, [out_a, mean_a, log_var_a, laten_a])

class ModelEncoderBN(BaseModel):
    def __init__(self, layers_conv=[], is_bn=False, dim_a=1024):

        self.layers = []
        if layers_conv:
            for filters in layers_conv:
                self.layers.append(Conv2D(filters, kernel_size=4, strides=2, padding='same'))
                if is_bn:
                    self.layers.append(BatchNormalization())

                self.layers.append(LeakyReLU(alpha=0.1))

        self.layers.append(Flatten())

        self.dense_out = keras.layers.Dense(units=dim_a)

    # init forward pass
    def init(self, inputs):

        n = len(self.layers)
        for i in range(n):
            if i == 0:
                x = self.layers[i](inputs)
            else:
                x = self.layers[i](x)

        out = self.dense_out(x)
        # out = self.act_out(out)

        return Model(inputs, out)

class ModelDecoderBN(BaseModel):
    def __init__(self, num_classes=10):

        self.dense1 = keras.layers.Dense(units=500)
        self.act1   = keras.layers.ReLU()

        self.dense_out = keras.layers.Dense(units=num_classes)
        self.act_out   = keras.layers.Softmax()

    # init forward pass
    def init(self, inputs):

        self.debug_print(inputs._keras_shape)

        x = self.dense1(inputs)
        x = self.act1(x)

        self.debug_print(x._keras_shape)

        x   = self.dense_out(x)
        out = self.act_out(x)

        self.debug_print(out._keras_shape)

        return Model(inputs, out)
