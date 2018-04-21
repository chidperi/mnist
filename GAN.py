# File name: CNN
# Copyright 2018 Chidambaram Periakaruppan


from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Conv2DTranspose


class Generator(object):
    def __init__(self, h, d, classes):
        self.trainable = True

        g_1 = Dense(128, activation='tanh', trainable=self.trainable, input_shape=(classes,))

        g_2 = Dense(200, activation='tanh', trainable=self.trainable)

        g_3 = Reshape(target_shape=(5, 5, 8))

        g_4 = Conv2DTranspose(8, (5, 5), strides=2, activation='relu', trainable=self.trainable)

        g_5 = Conv2DTranspose(8, (4, 4), strides=2, activation='relu', padding='valid', trainable=self.trainable)

        g_6 = Conv2DTranspose(4, (4, 4), strides=1, activation='relu', padding='same', trainable=self.trainable)

        g_7 = Conv2DTranspose(2, (4, 4), strides=1, activation='relu', padding='same', trainable=self.trainable)

        g_8 = Conv2DTranspose(1, (4, 4), strides=1, activation='relu', padding='same', trainable=self.trainable)

        self.model = Sequential([

            g_1,
            g_2,
            g_3,
            g_4,
            g_5,
            g_6,
            g_7,
            g_8,
        ])

    def frozen_model(self):
        self.trainable = False
        self.model.layers[0].trainable = self.trainable
        self.model.layers[1].trainable = self.trainable
        self.model.layers[2].trainable = self.trainable
        self.model.layers[3].trainable = self.trainable
        self.model.layers[4].trainable = self.trainable
        self.model.layers[5].trainable = self.trainable
        self.model.layers[6].trainable = self.trainable
        self.model.layers[7].trainable = self.trainable

    def trainable_model(self):
        self.trainable = True
        self.model.layers[0].trainable = self.trainable
        self.model.layers[1].trainable = self.trainable
        self.model.layers[2].trainable = self.trainable
        self.model.layers[3].trainable = self.trainable
        self.model.layers[4].trainable = self.trainable
        self.model.layers[5].trainable = self.trainable
        self.model.layers[6].trainable = self.trainable
        self.model.layers[7].trainable = self.trainable


class Discriminator(object):
    def __init__(self, h, d, classes):
        self.trainable = True
        #         d_0 = Input(shape=(h, d, 1))

        d_1 = Conv2D(2, (3, 3), strides=1, padding='same', activation='relu', trainable=self.trainable,
                     input_shape=(h, d, 1))

        d_2 = Conv2D(4, (3, 3), strides=1, padding='same', activation='relu', trainable=self.trainable)

        d_3 = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu', trainable=self.trainable)

        d_4 = Conv2D(8, (4, 4), strides=2, padding='valid', activation='relu', trainable=self.trainable)

        d_5 = Conv2D(8, (4, 4), strides=2, padding='valid', activation='relu', trainable=self.trainable)

        d_6 = Flatten()
        d_7 = Dense(128, activation='tanh', trainable=self.trainable)

        d_8 = Dense(classes, activation='softmax', trainable=self.trainable)

        self.model = Sequential([
            #             d_0,
            d_1,
            d_2,
            d_3,
            d_4,
            d_5,
            d_6,
            d_7,
            d_8,
        ])

    def frozen_model(self):
        self.trainable = False
        self.model.layers[0].trainable = self.trainable
        self.model.layers[1].trainable = self.trainable
        self.model.layers[2].trainable = self.trainable
        self.model.layers[3].trainable = self.trainable
        self.model.layers[4].trainable = self.trainable
        self.model.layers[5].trainable = self.trainable
        self.model.layers[6].trainable = self.trainable
        self.model.layers[7].trainable = self.trainable

    def trainable_model(self):
        self.trainable = True
        self.model.layers[0].trainable = self.trainable
        self.model.layers[1].trainable = self.trainable
        self.model.layers[2].trainable = self.trainable
        self.model.layers[3].trainable = self.trainable
        self.model.layers[4].trainable = self.trainable
        self.model.layers[5].trainable = self.trainable
        self.model.layers[6].trainable = self.trainable
        self.model.layers[7].trainable = self.trainable


# def Discriminator(Xin, classes, trainable = True):

#     d_1.trainable = trainable
#     d_2.trainable = trainable
#     d_3.trainable = trainable
#     d_4.trainable = trainable
#     d_5.trainable = trainable
#     d_6.trainable = trainable
#     d_7.trainable = trainable
#     d_8.trainable = trainable

#     X = d_1(Xin)
#     X = d_2(X)
#     X = d_3(X)
#     X = d_4(X)
#     X = d_5(X)
#     X = d_6(X)
#     X = d_7(X)
#     out = d_8(X)

#     return out


Xin = Input(shape=(10,))
generator = Generator(28, 28, 10)
X = generator.model(Xin)
discriminator = Discriminator(28, 28, 10)
Xout = discriminator.model(X)

out = Xout

model = Model(inputs=Xin, outputs=out)

model.summary()

def generate_noise(samples, data):
    noise = np.random.random((samples,data))*50
    softmax_noise = np.exp(noise)
    softmax_noise = softmax_noise/softmax_noise.sum(axis=1, keepdims=True)
    return softmax_noise

noise = generate_noise(100,10)

def generate_image_labels(generator, noise):
    generator.model.compile(optimizer='Adam',loss='categorical_crossentropy')
    generator_images = generator.model.predict(noise)
    generator_labels=np.argmax(noise, axis=1).reshape(-1,1)
    generate_images = mnist_data.X_denormalize(generator_images)

    return generator_images, generator_labels

generator_images, generator_labels = generate_image_labels(generator, noise)
mnist_data.show_image(images=generator_images, index=5, labels=generator_labels)


class GAN(object):
    def __init__(self):
        self.model = None

    def Discriminator(self, input_image):
        '''
        Initializes the convolutional neural network model for MNIST database.
        Args:
            h: height of the picture in pixels
            d: depth of the picture in pixels
            classes: no of classes

        '''


        Xin = Input(shape=(h, d, 1))

        X = Conv2D(2, (3, 3), strides=1, padding='same', activation='relu')(Xin)

        X = Conv2D(4, (3, 3), strides=1, padding='same', activation='relu')(X)

        X = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu')(X)

        X = Conv2D(8, (4, 4), strides=2, padding='valid', activation='relu')(X)

        X = Conv2D(8, (4, 4), strides=2, padding='valid', activation='relu')(X)

        X = Flatten()(X)
        X = Dense(128, activation='tanh')(X)

        X = Dense(classes, activation='softmax')(X)

        out = X

        self.model = Model(inputs=Xin, outputs=out)


    def Generator(h, d, classes):
        Xin = Input(shape=(10,))

        X = Dense(128, activation='tanh')(Xin)

        X = Dense(200, activation='tanh')(X)

        X = Reshape(target_shape=(5, 5, 8))(X)

        X = Conv2DTranspose(8, (5, 5), strides=2, activation='relu')(X)

        X = Conv2DTranspose(8, (4, 4), strides=2, activation='relu', padding='valid')(X)

        X = Conv2DTranspose(4, (4, 4), strides=1, activation='relu', padding='same')(X)

        X = Conv2DTranspose(2, (4, 4), strides=1, activation='relu', padding='same')(X)

        X = Conv2DTranspose(1, (4, 4), strides=1, activation='relu', padding='same')(X)

        out = X

        model = Model(inputs=Xin, outputs=out)
        return model



    def training_model(self, h, d, classes):
        '''
        Initializes the convolutional neural network model for MNIST database.
        Args:
            h: height of the picture in pixels
            d: depth of the picture in pixels
            classes: no of classes

        '''
        noise = Input(shape=(h, d, 1), name='noise')
        data = Input(shape=(h, d, 1), name='data')

        noiseout = self.Discriminator(noise)


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

if __name__ == '__main__':
    main()