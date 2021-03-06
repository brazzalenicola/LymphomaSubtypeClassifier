import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv2D, Input, Softmax
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Flatten
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray, rgb2hsv
import preprocessing
import utils

def CNN_model(input_shape):

    X_input = Input(input_shape)
    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv0')(X_input)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max_pool0')(X)
    X = Activation('relu')(X)

    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)

    X = AveragePooling2D(pool_size=(2, 2), strides=None, name='avg_pool0')(X)
    X = Conv2D(64, (5, 5), strides=(1, 1), name='conv2')(X)
    X = Activation('relu')(X)

    X = AveragePooling2D(pool_size=(2, 2), strides=None, name='avg_pool1')(X)
    X = Flatten()(X)


    X = Dense(64, activation='relu', name='fc0')(X)
    X = Dropout(0.5)(X)

    X = Dense(3, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Softmax()(X)

    CNNmodel = keras.Model(inputs=X_input, outputs=X, name='CNNmodel')


    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=5250, decay_rate=0.9)
    opt = keras.optimizers.SGD(learning_rate=lr_schedule)
    CNNmodel.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    print(CNNmodel.summary())
    return CNNmodel

def trainCNN(model, ep, color_space):

    #number of channels
    nch = 3
    if(color_space == "gray"):
        X_train, y_train = preprocessing.loadTrainingGraySet()
        nch = 1
    elif(color_space == "hsv"):
        X_train, y_train = preprocessing.loadTrainingHSVSet()
    elif(color_space == "yuv"):
        X_train, y_train = preprocessing.loadTrainingYUVSet()
    else:
        X_train, y_train = preprocessing.loadTrainingSet()


    X_train, y_train = preprocessing.extract_patches(X_train, y_train, nch)

    history = model.fit(X_train, y_train, epochs=ep, batch_size=32)
    utils.plot_Accuracy_Loss(history)
    model.save("CNN.h5")


def evaluateCNN(CNNmodel, color_space):

    nch = 3
    if(color_space == "gray"):
        X_test, y_test = preprocessing.loadTestGraySet()
        nch = 1
    elif(color_space == "hsv"):
        X_test, y_test = preprocessing.loadTestHSVSet()
    elif(color_space == "yuv"):
        X_test, y_test = preprocessing.loadTestYUVSet()
    else:
        X_test, y_test = preprocessing.loadTestSet()

    y_test_imgs = y_test
    X_test, y_test = preprocessing.extract_patches(X_test, y_test, nch)

    print("Patch-Wise Evaluation:")
    preds = CNNmodel.evaluate(x=X_test, y=y_test)

    print("Test Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    y_pred = []
    yPredictedProbs = CNNmodel.predict(X_test)
    yPredicted = yPredictedProbs.argmax(axis=1)
    yPredicted = np.reshape(yPredicted, (yPredicted.shape/336, 336))

    for pred in yPredicted:
        count_bins = np.bincount(pred)
        y_pred.append(np.argmax(count_bins))
    print("Image-Wise Evaluation: " + str((np.sum(y_test_imgs == y_pred) / 124) * 100) + " %")
    utils.print_confusion_matrix(y_test_imgs, y_pred, 3, ['FL', 'MCL', 'CLL'])



