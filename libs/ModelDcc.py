import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers import Dropout, BatchNormalization
from keras.layers import Activation, ReLU, LeakyReLU, Softmax

import sys

from libs.BaseModel import BaseModel
# ======================================================================================================================
class ModelDcc(BaseModel):
    def __init__(self, num_classes=10, layers_conv=[], layers_dense=[], is_bn=False):

        self.layers = []
        if layers_conv:
            for filters in layers_conv:
                self.layers.append(Conv2D(filters, kernel_size=4, strides=2, padding='same'))
                if is_bn:
                    self.layers.append(BatchNormalization())

                self.layers.append(LeakyReLU(alpha=0.1))

        self.layers.append(Flatten())

        if layers_dense:
            for filter_size in layers_dense:
                self.layers.append(Dense(units=filter_size))
                self.layers.append(ReLU())

        self.dense_out = keras.layers.Dense(units=num_classes)
        self.act_out = keras.layers.Softmax()

    # init forward pass
    def init(self, inputs):

        n = len(self.layers)
        for i in range(n):
            if i == 0:
                x = self.layers[i](inputs)
            else:
                x = self.layers[i](x)

        x = self.dense_out(x)
        out = self.act_out(x)

        return Model(inputs, out)