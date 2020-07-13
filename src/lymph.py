import tensorflow as tf
import tensorflow.keras as keras
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
import pickle
from tensorflow.keras import backend as K


# print(tf.__version__)
import IRRCNN


def extract_patches(X_set, y_lab):
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
    image_patches = tf.reshape(image_patches, [b * nr * nc, 64, 64, 3])
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

    #random.shuffle(all_data)

    X = []  # images
    y = []  # labels
    for images, label in all_data:
        X.append(images)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(y)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1238164)

    # Creating the files containing all the information about your model
    pickle_out = open("X_train.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    return _

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

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_dataset()

    X_train, y_train = extract_patches(X_train, y_train)
    X_test, _ = extract_patches(X_test, y_test)

    # Normalize image vectors
    X_train = X_train / 255
    X_test = X_test / 255

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

    #IRRCNN model

    image_size = 64
    no_classes = 3

    if K.image_data_format() == 'channels_first':
        inputs = tf.keras.layers.Input(shape=(3, image_size, image_size))
    else:
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    x = keras.layers.Conv2D(32, (3, 3), strides=(2,2), padding='valid')(inputs)
    x = IRRCNN.IRRCNN_model(x)
    x = keras.layers.Dense(units=no_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, x, name='IRRCNN')
    summ = model.summary()
    #print(summ)

    epochs = 1
    opt = keras.optimizers.SGD(lr = 1e-2)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer= opt, metrics=["accuracy"])
    #history = model.fit(x = X_train, y = y_train, epochs=epochs, batch_size=32)
    #model.save("IRRCNN.h5")

    '''
    plt.figure()
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['acc'], label='Train loss')
    plt.plot(history.history['val_acc'], label='Val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    '''
    testmodel = keras.models.load_model('/Users/brazzalenicola/Desktop/IRRCNN.h5')

    print(X_test.shape)

    y_pred = []

    print("Evaluating...")
    for i in range(1, X_test.shape[0], 336):
        print((i/336)/(X_test.shape[0]/336)*100 + "% Completed")

        #Prediction of the patches in a single original image (336 patches per image)
        yPredictedProbs = testmodel.predict(X_test[i:i+336, :, :, :])
        yMaxPredictedProbs = np.amax(yPredictedProbs, axis=1)
        yPredicted = yPredictedProbs.argmax(axis=1)
        count = np.bincount(yPredicted)
        y_pred.append(np.argmax(count))
    

    IRRCNN.print_confusion_matrix(y_test, y_pred, 3, ["CLL", "FL", "MCL"])



