import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Lambda
from keras.layers import Dropout, BatchNormalization
from keras.layers import Activation, ReLU, LeakyReLU, Softmax
from keras import backend as K

import sys

from libs.BaseModel import BaseModel
# ======================================================================================================================
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
