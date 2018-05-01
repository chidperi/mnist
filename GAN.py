# File name: CNN
# Copyright 2018 Chidambaram Periakaruppan


from keras import Input, Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Conv2DTranspose, Reshape
import numpy as np
import mnist_data


class Generator(object):
    '''

    '''

    def __init__(self, h, d, classes):
        self.trainable = True

        g_1 = Dense(100, activation='relu', trainable=self.trainable, input_shape=(classes,))

        g_2 = BatchNormalization()

        g_3 = Dense(4096, activation='relu', trainable=self.trainable)

        g_4 = BatchNormalization()

        g_5 = Reshape(target_shape=(2, 2, 1024))

        g_6 = Conv2DTranspose(512, (3, 3), strides=2, activation='relu', padding='valid', trainable=self.trainable)

        g_7 = BatchNormalization()

        g_8 = Conv2DTranspose(256, (3, 3), strides=1, activation='relu', padding='valid', trainable=self.trainable)

        g_9 = BatchNormalization()

        g_10 = Conv2DTranspose(128, (4, 4), strides=2, activation='relu', padding='same', trainable=self.trainable)

        g_11 = BatchNormalization()

        g_12 = Conv2DTranspose(1, (4, 4), strides=2, activation='tanh', padding='same', trainable=self.trainable)

        self.model = Sequential([

            g_1,
            g_2,
            g_3,
            g_4,
            g_5,
            g_6,
            g_7,
            g_8,
            g_9,
            g_10,
            g_11,
            g_12

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
        d_2 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')

        d_3 = Conv2D(4, (3, 3), strides=1, padding='same', activation='relu', trainable=self.trainable)
        d_4 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')

        d_5 = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu', trainable=self.trainable)
        d_6 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')

        d_7 = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu', trainable=self.trainable)
        d_8 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        d_9 = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu', trainable=self.trainable)
        d_10 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        d_11 = Flatten()
        d_12 = Dense(128, activation='tanh', trainable=self.trainable)

        d_13 = Dense(classes, activation='softmax', trainable=self.trainable)

        self.model = Sequential([

            d_1,
            d_2,
            d_3,
            d_4,
            d_5,
            d_6,
            d_7,
            d_8,
            d_9,
            d_10,
            d_11,
            d_12,
            d_13
        ])

    #         d_1 = Conv2D(2, (3, 3), strides=1, padding='same', activation='relu', trainable = self.trainable,input_shape=(h, d, 1))

    #         d_2 = Conv2D(4, (3, 3), strides=1, padding='same', activation='relu', trainable = self.trainable)

    #         d_3 = Conv2D(8, (3, 3), strides=1, padding='same', activation='relu', trainable = self.trainable)

    #         d_4 = Conv2D(8, (4, 4), strides=2, padding='valid', activation='relu', trainable = self.trainable)

    #         d_5 = Conv2D(8, (4, 4), strides=2, padding='valid', activation='relu', trainable = self.trainable)

    #         d_6 = Flatten()
    #         d_7 = Dense(128, activation='tanh', trainable = self.trainable)

    #         d_8 = Dense(classes, activation='softmax', trainable = self.trainable)

    #         self.model = Sequential([
    # #             d_0,
    #             d_1,
    #             d_2,
    #             d_3,
    #             d_4,
    #             d_5,
    #             d_6,
    #             d_7,
    #             d_8,
    #         ])

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
        self.model.layers[8].trainable = self.trainable
        self.model.layers[9].trainable = self.trainable
        self.model.layers[10].trainable = self.trainable
        self.model.layers[11].trainable = self.trainable
        self.model.layers[12].trainable = self.trainable

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
        self.model.layers[8].trainable = self.trainable
        self.model.layers[9].trainable = self.trainable
        self.model.layers[10].trainable = self.trainable
        self.model.layers[11].trainable = self.trainable
        self.model.layers[12].trainable = self.trainable


class GAN(object):
    def __init__(self, w, d, classes,loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']):
        self.generator = Generator(w, d, classes)
        self.discriminator = Discriminator(w, d, classes + 1)

        generated_data = Input(shape=(classes,))
        gen_X = self.generator.model(generated_data)
        gen_X = self.discriminator.model(gen_X)
        gen_out = gen_X

        actual_data = Input(shape=(w, d, 1))
        act_X = self.discriminator.model(actual_data)
        act_out = act_X

        self.generator.frozen_model()
        self.discriminator.trainable_model()
        self.discriminator_trainer = Model(inputs=[generated_data, actual_data], outputs=[gen_out, act_out])
        self.discriminator_trainer.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.generator.trainable_model()
        self.discriminator.frozen_model()
        self.generator_trainer = Model(inputs=[generated_data], outputs=[gen_out])
        self.generator_trainer.compile(loss=loss, optimizer=optimizer, metrics=metrics)


    def train(self, softmax_noise, discriminator_training_labels_oh,generator_training_labels_oh,
              actual_images, training_labels_oh):

        samples = softmax_noise.shape[0]
        random = list(range(samples))
        np.random.shuffle(random)
        for i in range(samples):

            random_batch = random[i:(i + 1)]
            random_batch_gen = random[i:(i + 1)]

            self.discriminator_trainer.train_on_batch(x=[softmax_noise[random_batch], actual_images[random_batch]],
                                                       y=[discriminator_training_labels_oh[random_batch],
                                                          training_labels_oh[random_batch]])

            self.generator_trainer.train_on_batch(x=[softmax_noise[random_batch_gen]],
                                                   y=[generator_training_labels_oh[random_batch_gen]])
            if i % 1000 == 0:
                print('iteration %s' % i)
                # self.get_generator_prediction(softmax_noise, np.random.randint(60000))


    def evaluate(self, test_images, test_labels_oh):
        '''
        Returns the loss and accuracy of the GAN Discriminator on the test set.
        Args:
            test_images:
            test_labels_oh:

        Returns:

        '''

        self.discriminator.model.evaluate(x=test_images, y=test_labels_oh)

    def get_generator_prediction(self, noise, index):

        gen_image = self.generator.model.predict(x=noise[index:index + 1])
        gen_image = mnist_data.X_denormalize(gen_image)
        labels = np.argmax(noise[index:index+1], axis=1).reshape(-1,1)

        get_prediction(self.discriminator, gen_image, labels, 0)

def generate_noise(samples, classes):
    '''
    Generates of samples of softmax noise for a given number of classes.

    Args:
        samples(int): number of samples to generate
        classes(int): number of classes

    Returns:
        An arry of shape(samples, classes) where each row conforms to softmax.

    '''

    noise = np.random.random((samples,classes))*50
    softmax_noise = np.exp(noise)
    softmax_noise = softmax_noise/softmax_noise.sum(axis=1, keepdims=True)
    return softmax_noise



def generate_image_labels(gan, softmax_noise):
    '''
    Producers an array of images generated by the Generator from the softmax_noise.
    Args:
        gan(Model): A keras Model
        softmax_noise(np.array): an of softmax outputs.

    Returns:

    '''
    gan.generator.model.compile(optimizer='Adam',loss='categorical_crossentropy')
    generator_images = gan.generator.model.predict(softmax_noise)
    generator_labels=np.argmax(softmax_noise, axis=1).reshape(-1,1)
    generator_images = mnist_data.X_denormalize(generator_images)

    return generator_images, generator_labels


def get_gan_onehot(test_labels, training_labels, generator_labels, classes):
    '''
    Given a test, training and generator labels, returns one hot representations
    that include an extra classification as to whether the sample is a known fake or not
    for training the discriminator and generator.

    Args:
        test_labels(np.array): actual data test labels
        training_labels(np.array): actual data training labels
        generator_labels(np.array): generator generated data labels
        classes(int): number of classes in ground truth

    Returns:
        test_labels_oh(np.array) : one hot test labels
        training_labels_oh(np.array): one hot training lables
        generator_training_labels_oh(np.array): one hot generator training lables
        discriminator_training_labels_oh(np.array): one hot discriminator training labels

    '''
    training_labels_oh = mnist_data.Y_onehot(training_labels, classes + 1)
    test_labels_oh = mnist_data.Y_onehot(test_labels, classes + 1)
    generator_training_labels_oh = mnist_data.Y_onehot(generator_labels, classes + 1)
    discriminator_training_labels_oh = np.zeros((generator_training_labels_oh.shape))
    discriminator_training_labels_oh[:, classes] = 1
    return test_labels_oh, training_labels_oh, generator_training_labels_oh, discriminator_training_labels_oh



def get_prediction(discriminator, images, labels, index):
    '''
    Takes a given GAN, images and labels and one index value to show what the image, the ground truth and the
    GAN prediction by the discriminator.
    Args:
        gan:
        images:
        labels:
        index:

    Returns:

    '''
    mnist_data.show_image(images=images, labels=labels, index=index)
    pred = np.argmax(discriminator.model.predict(x=images[index:index + 1, :, :, :]), axis=1)[0]
    print('Prediction is %s' % pred)





def main():
    import mnist_data
    training_images, training_labels, test_images, test_labels = mnist_data.load()
    training_images, training_labels_oh, test_images, test_labels_oh = mnist_data.get_data_for_model(training_images,
                                                                                                     training_labels,
                                                                                                     test_images,
                                                                                                     test_labels, 10)


    model = GAN(28, 28, 10)
    softmax_noise = generate_noise(60000, 10)
    generator_images, generator_labels = generate_image_labels(model, softmax_noise)
    test_labels_oh, training_labels_oh, generator_training_labels_oh, discriminator_training_labels_oh = get_gan_onehot(
        test_labels, training_labels, generator_labels, 10)

    model.train(softmax_noise, discriminator_training_labels_oh,generator_training_labels_oh,
              training_images, training_labels_oh)
    return model, softmax_noise, generator_labels

if __name__ == '__main__':
    main()