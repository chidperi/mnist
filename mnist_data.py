# File name: mnist_data
# Copyright 2018 Chidambaram Periakaruppan 

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


def load(directory='./data/'):
    '''
    Uses python-mnist package to load images and labels for training and testing.
    Args:
        directory(str): location of unzipped mnist data

    Returns:
        training_images, training_labels, test_images, test_labels
    '''

    mndata = MNIST(directory)
    training_images, training_labels = list_to_array(*mndata.load_training())
    test_images, test_labels = list_to_array(*mndata.load_testing())
    return training_images, training_labels, test_images, test_labels


def show_image(index, images, labels):
    '''
    Shows image and label from a images and labels for a given index.
    Args:
        index(int):
        images(np.array):
        labels(np.array):

    Returns:


    '''

    plt.imshow(images[index, :, :, 0], cmap='gray_r')
    print('Label is %s' % labels[index, 0])
    plt.show()


def list_to_array(images, labels):
    '''
    Convert the lists of images and labels into array of images and labels.
    Args:
        images: list of 1d lists to covert to inputs
        labels: list of labels to conver to (m,1) array

    Returns:
        X: mx28x28 array of images.
        Y: mx1 arry of labels

    '''
    m = len(images)
    X = np.array(images).reshape(m, 28, 28, 1)
    Y = np.array(labels).reshape(m, 1)
    return X, Y


def X_normalize(train, test, minval=0., maxval=255.):
    '''
    Normalize X values by assuming mean is 0 and dividing by maxval-minval.
    Args:
        train(np.array): train X data
        test(np.array): test X data
        minval(float): minimum value
        maxval(float): maximum value

    Returns:

    '''
    minval = float(minval)
    maxval = float(maxval)
    mean = 0.5 * (maxval - minval)
    train = (train - mean) / mean
    test = (test - mean) / mean
    return train, test


def X_denormalize(X, minval=0., maxval=255.):
    '''
    Denormalize X values by assuming mean is 0 and dividing by maxval-minval.
    Args:
        X(np.array):  X data
        minval(float): minimum value
        maxval(float): maximum value

    Returns:
        X_denorm: denormalized X data back

    '''
    minval = float(minval)
    maxval = float(maxval)
    mean = 0.5 * (maxval - minval)
    X_denorm = (X * mean) + mean

    return X_denorm


def Y_onehot(Y, classes=10):
    '''
    Encode the Y labels into a one hot array.
    Args:
        Y(np.array): labels
        classes(int): total number of classes

    Returns:
        Yoh(np.array): one-hot encoded Y array.
    '''
    m, _ = Y.shape
    Yoh = np.zeros((m, classes))
    Yoh[np.arange(m), Y[:, 0]] = 1

    return Yoh


def Y_labels(Yoh):
    '''
    Returns labels array for the one-hot encoded array
    Args:
        Yoh(np.array): one-hot encoded array of labels

    Returns:
        Y(np.array): array of labels

    '''
    return np.argmax(Yoh, axis=1)


def get_data_for_model(training_images, training_labels, test_images, test_labels, classes):
    training_images, test_images = X_normalize(training_images, test_images)
    training_labels_oh = Y_onehot(training_labels, classes)
    test_labels_oh = Y_onehot(test_labels, classes)

    return training_images, training_labels_oh, test_images, test_labels_oh


if __name__ == '__main__':
    training_images, training_labels, test_images, test_labels = load()
    show_image(2, training_images, training_labels)
