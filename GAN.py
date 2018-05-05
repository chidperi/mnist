# File name: CNN
# Copyright 2018 Chidambaram Periakaruppan


import numpy as np
from keras import Input, Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose, Reshape, BatchNormalization, Activation
from keras.layers import LeakyReLU
import mnist_data


class Generator(object):
    '''

    '''

    def __init__(self, h, d, classes):


        g_1 = Dense(2048, input_shape=(classes,), name='G_0_1')

#         g_2 = BatchNormalization()

#         g_3 = Activation(activation='relu')

#         g_4 = Dense(128)

# #         g_5 = BatchNormalization()

#         g_6 = Activation(activation='relu')

        g_7 = Reshape(target_shape=(2, 2, 512))

        g_8 = Conv2DTranspose(256, (3, 3), strides=2, padding = 'valid')

        g_9 = BatchNormalization()

        g_10 = Activation(activation='relu')

        g_11 = Conv2DTranspose(128, (3, 3), strides=1, padding='valid')

        g_12 = BatchNormalization()

        g_13 = Activation(activation='relu')

        g_14 = Conv2DTranspose(64, (4, 4), strides=2, padding='same')

        g_15 = BatchNormalization()

        g_16 = Activation(activation='relu')

        g_17 = Conv2DTranspose(1, (4, 4), strides=2, activation='tanh', padding='same')

        self.model = Sequential([

            g_1,
# #             g_2,
#             g_3,
#             g_4,
# #             g_5,
#             g_6,
            g_7,
            g_8,
            g_9,
            g_10,
            g_11,
            g_12,
            g_13,
            g_14,
            g_15,
            g_16,
            g_17,

        ])



class Discriminator(object):
    def __init__(self, h, d, classes):



        d_1 = Conv2D(64, (4, 4), strides=2, padding='same',
                     input_shape=(h, d, 1))
        d_2 = BatchNormalization()
        d_3 = LeakyReLU(0.2)
        d_4 = Conv2D(128, (4, 4), strides=2, padding='same')
        d_5 = BatchNormalization()
        d_6 = LeakyReLU(0.2)
        d_7 = Conv2D(256, (3, 3), strides=1, padding='valid')
        d_8 = BatchNormalization()
        d_9 = LeakyReLU(0.2)
        d_10 = Conv2D(512, (3, 3), strides=2, padding='valid')
        d_11 = BatchNormalization()
        d_12 = LeakyReLU(0.2)
        d_13 = Flatten()
        d_14 = Dense(100)
        d_15 = LeakyReLU(0.2)
        d_16 = Dense(classes, activation='softmax')



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
            d_13,
            d_14,
            d_15,
            d_16,

        ])



from keras.optimizers import Adam
class GAN(object):
    def __init__(self, w, d, classes,loss='categorical_crossentropy',
                 optimizer=Adam(lr=0.0002,beta_1=0.5), metrics=['accuracy']):
        self.generator = Generator(w, d, classes)
        self.discriminator = Discriminator(w, d, classes + 1)

        generated_data = Input(shape=(classes,))
        gen_X = self.generator.model(generated_data)
        gen_X = self.discriminator.model(gen_X)
        gen_out = gen_X

        actual_data = Input(shape=(w, d, 1))
        act_X = self.discriminator.model(actual_data)
        act_out = act_X

        self.generator.model.trainable = False
        self.discriminator.model.trainable = True
        self.discriminator_trainer = Model(inputs=[generated_data, actual_data], outputs=[gen_out, act_out])
        self.discriminator_trainer.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.generator.model.trainable = True
        self.discriminator.model.trainable = False

        self.generator_trainer = Model(inputs=[generated_data], outputs=[gen_out])
        self.generator_trainer.compile(loss=loss, optimizer=optimizer, metrics=metrics)


    def train(self, softmax_noise, discriminator_training_labels_oh,generator_training_labels_oh,
              actual_images, training_labels_oh):

        samples = softmax_noise.shape[0]
        random = list(range(samples))
        np.random.shuffle(random)
        for j in range(1):
            for i in range(1000):

                random_batch = random[i:(i + 1)]


                self.discriminator_trainer.train_on_batch(x=[softmax_noise[random_batch], actual_images[random_batch]],
                                                           y=[discriminator_training_labels_oh[random_batch],
                                                              training_labels_oh[random_batch]])

                self.generator_trainer.train_on_batch(x=[softmax_noise[random_batch]],
                                                       y=[generator_training_labels_oh[random_batch]])
                if i % 1000 == 0:
                    print('iteration %s of %s' % (i,j))
                    print(self.discriminator_trainer.evaluate(x=[softmax_noise[:128], actual_images[:128]],
                                                           y=[discriminator_training_labels_oh[:128],
                                                              training_labels_oh[:128]]))

                    self.get_generator_prediction(softmax_noise, np.random.randint(60000))


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



def generate_labels(softmax_noise):
    '''
    Producers an array of images generated by the Generator from the softmax_noise.
    Args:
        softmax_noise(np.array): an of softmax outputs.

    Returns:

    '''
    # gan.generator.model.compile(optimizer='Adam',loss='categorical_crossentropy')
    # generator_images = gan.generator.model.predict(softmax_noise)
    generator_labels=np.argmax(softmax_noise, axis=1).reshape(-1,1)
    # generator_images = mnist_data.X_denormalize(generator_images)

    return generator_labels


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
    denorm_images = mnist_data.X_denormalize(images)
    mnist_data.show_image(images=denorm_images, labels=labels, index=index)
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
    generator_labels = generate_labels(softmax_noise)
    test_labels_oh, training_labels_oh, generator_training_labels_oh, discriminator_training_labels_oh = get_gan_onehot(
        test_labels, training_labels, generator_labels, 10)

    model.train(softmax_noise, discriminator_training_labels_oh,generator_training_labels_oh,
              training_images, training_labels_oh)

    model.discriminator_trainer.save('/home/chidperi/Projects/mnist/gan_model.h5')
    return model, softmax_noise, generator_labels

if __name__ == '__main__':
    main()