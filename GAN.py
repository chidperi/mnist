# File name: GAN
# Copyright 2018 Chidambaram Periakaruppan 

from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten


class NN(object):
    def __init__(self):
        self.model = None

    def CNN(self, h, d, classes):
        '''
        Initializes the convolutional neural network model for MNIST database.
        Args:
            h: height of the picture in pixels
            d: depth of the picture in pixels
            classes: no of classes

        '''
        Xin = Input(shape=(h, d, 1))

        X = Conv2D(2, (3, 3), strides=1, padding='same', activation='relu')(Xin)
        X = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(X)

        X = Conv2D(4, (3, 3), strides=1, padding='same', activation='relu')(X)
        X = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(X)

        X = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu')(X)
        X = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(X)

        X = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu')(X)
        X = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(X)

        X = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu')(X)
        X = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(X)

        X = Flatten()(X)
        X = Dense(128, activation='tanh')(X)

        X = Dense(classes, activation='softmax')(X)

        out = X

        self.model = Model(inputs=Xin, outputs=out)
