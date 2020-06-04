import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers import Dropout, BatchNormalization
from keras.layers import Activation, ReLU, LeakyReLU, Softmax

from libs.BaseModel import BaseModel
# ======================================================================================================================

class ModelDx(BaseModel):
    def __init__(self, filters = [64, 64, 128, 256], kernel_size = 5):

        self.layers = []
        for filter in filters:
            # first 3 convolution layers use strides = 2
            # last one uses strides = 1
            if filter == filters[-1]:
                strides = 1
            else:
                strides = 2
            self.layers.append(Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding='same'))
            self.layers.append(LeakyReLU(alpha=0.2))

        self.layers.append(Flatten())


        self.layers.append(Dense(units=1))
        self.layers.append(Activation('sigmoid'))

    # init forward pass
    def init(self, inputs):

        n = len(self.layers)
        for i in range(n):
            if i == 0:
                x = self.layers[i](inputs)
            else:
                x = self.layers[i](x)

        return Model(inputs, x)