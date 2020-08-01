import tensorflow as tf
# Helper libraries
import numpy as np
from sklearn.model_selection import train_test_split

import os
import pathlib
import cv2
import pickle


def extract_patches(X_set, y_lab, n_ch):
    """
    Args:
        images set: numpy array containing the images to be processed to extract patches
        labels: numpy array containing the corresponding labels of the images set
    Return:
        patches: numpy array containing all the patches for each image
        labels: numpy array containing the labels for each patch
    """
    # Patches of 64x64 size
    ksizes = [1, 64, 64, 1]
    # We don't want overlapping patches, so the distance between the two centers is 64
    strides = [1, 64, 64, 1]
    # sample pixel consecutively
    rates = [1, 1, 1, 1]
    # only patches which are fully contained in the input image are included
    pad = 'VALID'

    image_patches = tf.image.extract_patches(X_set, ksizes, strides, rates, pad)
    print(image_patches.shape)
    b, nr, nc, d = image_patches.shape
    image_patches = tf.reshape(image_patches, [b * nr * nc, 64, 64, n_ch])
    X = image_patches
    y = np.repeat(y_lab, nr * nc)
    return X, y


def create_dataset(path):
    """
    Args:
        path: string, path to the folder containing the downloaded datasets
    Return:
        train_images, test_images: numpy tensors containing
                the training and test images
                [image_num, height, width, channels]
        train_labels, test_labels: list of int, they containing the correct class indices for
                                    each of the images in train_images, test_images
    """
    data_dir = pathlib.Path(path)
    IMG_SIZE = len(list(data_dir.glob('*/*.tif')))
    CLASSES = list([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])

    print(CLASSES)
    # print(IMG_SIZE)
    all_data = []

    for CLASS in CLASSES:
        path = os.path.join(data_dir, CLASS)
        class_num = CLASSES.index(CLASS)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (1344, 1024))
                print(class_num)
                all_data.append([new_array, class_num])
            except Exception as e:
                pass

    X = []  # images
    y = []  # labels
    for images, label in all_data:
        X.append(images)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1238164)

    # Creating the files containing all the information about your model
    '''
    pickle_out = open("X_train.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()
    '''
    return X_train, X_test, y_train, y_test

def create_gray_dataset():
    X_train, X_test, y_train, y_test = load_dataset()
    X_train = tf.image.rgb_to_grayscale(X_train)
    X_test = tf.image.rgb_to_grayscale(X_test)

    pickle_out = open("X_train_gray.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test_gray.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

def create_hsv_dataset():
    X_train, X_test, _, _ = load_dataset()

    X_train = X_train / 255
    X_test = X_test / 255
    X_train = tf.image.rgb_to_hsv(X_train)
    X_test = tf.image.rgb_to_hsv(X_test)

    pickle_out = open("X_train_hsv.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test_hsv.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

def create_yuv_dataset():
    X_train, X_test, _, _ = load_dataset()

    X_train = X_train / 255
    X_test = X_test / 255
    X_train = tf.image.rgb_to_yuv(X_train)
    X_test = tf.image.rgb_to_yuv(X_test)

    pickle_out = open("X_train_yuv.pickle", "wb")
    pickle.dump(X_train, pickle_out, protocol=4)
    pickle_out.close()

    pickle_out = open("X_test_yuv.pickle", "wb")
    pickle.dump(X_test, pickle_out, protocol=4)
    pickle_out.close()

def load_dataset():
    pickle_in = open("X_train.pickle", "rb")
    X_train = pickle.load(pickle_in)

    pickle_in = open("X_test.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle", "rb")
    y_train = pickle.load(pickle_in)

    pickle_in = open("y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)

    return X_train, X_test, y_train, y_test

def loadTrainingSet():
    pickle_in = open("X_train.pickle", "rb")
    X_train = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle", "rb")
    y_train = pickle.load(pickle_in)

    '''
    n_train = X_train.shape[0]
    print("number of training examples = " + str(n_train))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    '''
    return X_train, y_train

def loadTrainingGraySet():
    pickle_in = open("X_train_gray.pickle", "rb")
    X_train = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle", "rb")
    y_train = pickle.load(pickle_in)

    '''
    n_train = X_train.shape[0]
    print("number of training examples = " + str(n_train))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    '''
    return X_train, y_train

def loadTestGraySet():
    pickle_in = open("X_test_gray.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)

    return X_test, y_test

def loadTrainingGraySet():
    pickle_in = open("X_train_hsv.pickle", "rb")
    X_train = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle", "rb")
    y_train = pickle.load(pickle_in)

    return X_train, y_train

def loadTestHSVSet():
    pickle_in = open("X_test_hsv.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)

    return X_test, y_test

def loadTrainingYUVSet():
    pickle_in = open("X_train_yuv.pickle", "rb")
    X_train = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle", "rb")
    y_train = pickle.load(pickle_in)

    return X_train, y_train
def loadTestYUVSet():

    pickle_in = open("X_test_yuv.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)

    return X_test, y_test



def loadTestSet():
    pickle_in = open("X_test.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)

    print("number of test examples = " + str(X_test.shape[0]))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))

    return X_test, y_test
