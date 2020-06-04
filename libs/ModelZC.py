import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers import Dropout, BatchNormalization
from keras.layers import Activation, ReLU, LeakyReLU, Softmax

import sys

from libs.BaseModel import BaseModel
# ======================================================================================================================
# ----- AAE --------------------------------
class ModelEncoderZC(BaseModel):
    def __init__(self, layers_conv=[], layers_dense=[],  is_bn=False, dim_z=1024, num_classes=10):

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


        # --- Z ---------
        self.layers_z = keras.layers.Dense(units=dim_z)

        # --- C(class) -------
        self.layers_c = []

        self.layers_c.append(Dense(units=num_classes))
        self.layers_c.append(Softmax())


    # init forward pass
    def init(self, inputs):
        n = len(self.layers)
        for i in range(n):
            if i == 0:
                x = self.layers[i](inputs)
            else:
                x = self.layers[i](x)

        # --- Z
        z = self.layers_z(x)

        # --- C (class)
        n = len(self.layers_c)
        for i in range(n):
            if i == 0:
                c = self.layers_c[i](x)
            else:
                c = self.layers_c[i](c)

        return Model(inputs, [z, c])

class ModelReconstructorZC(BaseModel):
    def __init__(self, input_layers=128, layers_deconv=[128, 128, 64, 1], image_size=28):

        image_resize = image_size // 4

        self.layers = []

        self.layers.append(Dense(image_resize * image_resize * input_layers))
        self.layers.append(Reshape((image_resize, image_resize, input_layers)))

        for filters in layers_deconv:
            # first two convolution layers use strides = 2
            # the last two use strides = 1
            if filters > layers_deconv[-2]:
                strides = 2
            else:
                strides = 1
            self.layers.append(BatchNormalization())
            self.layers.append(ReLU())
            self.layers.append(Conv2DTranspose(filters=filters, kernel_size=5, strides=strides, padding='same'))

        self.layers.append(keras.layers.Activation('sigmoid'))

    # init forward pass
    def init(self, inputs):

        n = len(self.layers)
        for i in range(n):
            if i == 0:
                x = self.layers[i](inputs)
            else:
                x = self.layers[i](x)

        return Model(inputs, x)

