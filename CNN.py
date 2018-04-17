# File name: CNN
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

    def DNN(self, h, d, classes):
        '''
        Initializes the deep neural network model for MNIST database.
        Args:
            h: height of the picture in pixels
            d: depth of the picture in pixels
            classes: no of classes


        '''
        Xin = Input(shape=(h, d, 1))

        X = Flatten()(Xin)
        X = Dense(512, activation='tanh')(X)
        X = Dense(256, activation='tanh')(X)
        X = Dense(128, activation='tanh')(X)

        X = Dense(64, activation='tanh')(X)
        X = Dense(32, activation='tanh')(X)
        X = Dense(16, activation='tanh')(X)
        X = Dense(classes, activation='softmax')(X)

        out = X

        self.model = Model(inputs=Xin, outputs=out)

    def CNN_model(self, X, Yoh, batch_size=1000, epochs=1, validation_split=0.1):
        m, h, d, _ = X.shape
        m, classes = Yoh.shape

        self.CNN(h, d, classes)

        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        self.model.fit(X, Yoh, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def main():
    import mnist_data
    training_images, training_labels, test_images, test_labels = mnist_data.load()
    training_images, training_labels_oh, test_images, test_labels_oh = mnist_data.get_data_for_model(training_images,
                                                                                                     training_labels,
                                                                                                     test_images,
                                                                                                     test_labels, 10)

    model = NN()
    model.CNN_model(training_images, training_labels_oh)

    model.model.evaluate(test_images, test_labels_oh)

    return model
