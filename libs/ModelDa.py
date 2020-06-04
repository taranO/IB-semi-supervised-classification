import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers import Dropout, BatchNormalization
from keras.layers import Activation, ReLU, LeakyReLU, Softmax

import sys

from libs.BaseModel import BaseModel
# ======================================================================================================================

class ModelDa(BaseModel):
    def __init__(self):

        self.dense1 = keras.layers.Dense(units=500)
        self.act1   = keras.layers.ReLU()

        self.dense2 = keras.layers.Dense(units=500)
        self.act2   = keras.layers.ReLU()

        self.dense_out = keras.layers.Dense(units=1)
        self.act_out   = keras.layers.Activation('sigmoid')

    # init forward pass
    def init(self, inputs):

        self.debug_print(inputs._keras_shape)

        x = self.dense1(inputs)
        x = self.act1(x)

        self.debug_print(x._keras_shape)

        x = self.dense2(x)
        x = self.act2(x)

        self.debug_print(x._keras_shape)

        x   = self.dense_out(x)
        out = self.act_out(x)

        self.debug_print(out._keras_shape)

        return Model(inputs, out)
