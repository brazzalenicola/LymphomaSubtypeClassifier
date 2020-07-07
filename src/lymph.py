import tensorflow as tf
import tensorflow.keras as keras

# from keras.models import Sequential
# from keras import layers
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
# from keras.models import Model
# Helper libraries
import numpy as np
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pathlib
import cv2
import random


# print(tf.__version__)
def extract_patches(X_set, y_lab):
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
    image_patches = tf.reshape(image_patches, [b * nr * nc, 64, 64, 3])
    X = image_patches
    y = np.repeat(y_lab, nr * nc)
    return X, y


def load_dataset(path):
    data_dir = pathlib.Path(path)
    IMG_SIZE = len(list(data_dir.glob('*/*.tif')))
    CLASSES = list([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])

    # print(CLASSES)
    # print(IMG_SIZE)
    all_data = []

    for CLASS in CLASSES:
        path = os.path.join(data_dir, CLASS)
        class_num = CLASSES.index(CLASS)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (1344, 1024))
                all_data.append([new_array, class_num])
            except Exception as e:
                pass

    random.shuffle(all_data)

    X = []  # images
    y = []  # labels
    for images, label in all_data:
        X.append(images)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1238164)
    '''
    # Creating the files containing all the information about your model
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)
    '''
    X_train, y_train = extract_patches(X_train, y_train)
    X_test, y_test = extract_patches(X_test, y_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset('/Users/brazzalenicola/Desktop/lymphoma/')

    # Normalize image vectors
    # X_train = X_train / 255
    # X_test = X_test / 255

    # Reshape
    y_train = y_train.T
    y_test = y_test.T

    n_train = X_train.shape[0]
    print("number of training examples = " + str(n_train))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))
